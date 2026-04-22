#!/usr/bin/env python3
"""
iCloud Shared Album -> Gemini Image Restyler -> Google Photos Uploader

1. Picks a random photo from a public iCloud shared album.
2. Restyls it with Gemini (image-generation model).
3. Saves to /output and uploads to a Google Photos shared album.

Setup:
    - export GEMINI_API_KEY="your_key"
    - place client_secret.json (OAuth2 Desktop App) in this directory
    - first run opens a browser for Google sign-in; token cached in token.json
"""

from PIL import Image
import requests
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google.genai import types
from google import genai
from pathlib import Path
from io import BytesIO
import uuid
import re
import json
import os
import random

from dotenv import load_dotenv
load_dotenv()


# -- Constants -----------------------------------------------------------------

ICLOUD_ALBUM_TOKEN = os.environ["ICLOUD_ALBUM_TOKEN"]
PHOTOS_SHARE_URL = os.environ.get("PHOTOS_SHARE_URL", "")
OUTPUT_DIR = Path("output")
_secret_candidates = sorted(Path(".").glob("client_secret*.json"))
CLIENT_SECRET_PATH = _secret_candidates[0] if _secret_candidates else Path(
    "client_secret.json")
TOKEN_PATH = Path("token.json")

GEMINI_MODEL = "gemini-2.5-flash-image"

_PHOTOS_SCOPES = [
    "https://www.googleapis.com/auth/photoslibrary.appendonly",
]

ALBUM_ID_CACHE = Path(".photos_album_id")

ART_STYLES = [
    "Ghibli", "Disney", "Spider-Verse", "Arcane", "90s Retro Anime",
    "Makoto Shinkai", "Cyberpunk", "Isometric Pixel Art", "Low-Poly",
    "Papercut", "Ligne Claire", "Ukiyo-e", "Mid-Century Modern",
    "Tim Burton Gothic", "Art Nouveau", "Art Deco", "Impressionism",
    "Chiaroscuro", "Pointillism", "Fauvism", "Risograph", "Grisaille",
]

_ICLOUD_HEADERS = {
    "Origin": "https://www.icloud.com",
    "Content-Type": "application/json",
}

# -- iCloud public shared-album helpers ----------------------------------------


def _fetch_photos(token: str) -> tuple[list[dict], str]:
    """Return (photos metadata list, resolved base URL)."""
    base = "https://p06-sharedstreams.icloud.com"

    for attempt in range(3):
        url = f"{base}/{token}/sharedstreams/webstream"
        try:
            r = requests.post(
                url, json={"streamCtag": None}, headers=_ICLOUD_HEADERS, timeout=45
            )
        except requests.exceptions.Timeout:
            if attempt == 2:
                raise
            print(
                f"   iCloud request timed out (attempt {attempt + 1}), retrying...")
            continue

        if r.status_code == 330:
            host = r.json().get("X-Apple-MMe-Host", "")
            base = f"https://{host}"
            continue

        r.raise_for_status()
        return r.json().get("photos", []), base

    raise RuntimeError(
        "Could not reach iCloud sharedstreams after 3 attempts.")


def _fetch_asset_urls(base: str, token: str, guids: list[str]) -> dict:
    r = requests.post(
        f"{base}/{token}/sharedstreams/webasseturls",
        json={"photoGuids": guids},
        headers=_ICLOUD_HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    return r.json().get("items", {})


def download_random_photo(token: str) -> tuple[bytes, str]:
    photos, base = _fetch_photos(token)
    if not photos:
        raise RuntimeError("Album is empty or inaccessible.")

    photo = random.choice(photos)
    guid = photo["photoGuid"]
    derivatives = photo.get("derivatives", {})

    if not derivatives:
        raise RuntimeError(f"Photo {guid} has no derivatives.")

    best_derivative = max(derivatives.values(),
                          key=lambda d: int(d.get("width", 0)))
    best_checksum = best_derivative.get("checksum")
    if not best_checksum:
        raise RuntimeError(f"No checksum in derivatives for photo {guid}.")

    asset_items = _fetch_asset_urls(base, token, [guid])
    info = asset_items.get(best_checksum)
    if not info:
        raise RuntimeError(f"CDN URL missing for checksum {best_checksum}.")

    img_url = f"https://{info['url_location']}{info['url_path']}"
    r = requests.get(img_url, timeout=60)
    r.raise_for_status()
    return r.content, guid


# -- Gemini restyling ----------------------------------------------------------


def _to_jpeg(image_bytes: bytes) -> bytes:
    buf = BytesIO()
    with Image.open(BytesIO(image_bytes)) as img:
        img.convert("RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def restyle_with_gemini(image_bytes: bytes, api_key: str) -> tuple[bytes, str]:
    """Returns (image_bytes, ext). Raises RuntimeError if blocked."""
    client = genai.Client(api_key=api_key)
    jpeg_bytes = _to_jpeg(image_bytes)
    prompt = Path("prompt.txt").read_text(encoding="utf-8").strip()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt, types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")],
        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
    )
    candidate = response.candidates[0] if response.candidates else None
    if candidate is None or candidate.content is None:
        reason = getattr(getattr(candidate, "finish_reason", None), "name", "UNKNOWN") if candidate else "NO_CANDIDATE"
        raise RuntimeError(f"Gemini blocked: {reason}")

    for part in candidate.content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            ext = part.inline_data.mime_type.split("/")[-1]
            return part.inline_data.data, "jpg" if ext == "jpeg" else ext

    raise RuntimeError("Gemini returned no image part.")


# -- Google Photos upload ------------------------------------------------------


def _auth_photos() -> Credentials:
    """OAuth2 flow with token caching. Opens browser on first run."""
    if not CLIENT_SECRET_PATH.exists():
        raise FileNotFoundError(
            f"{CLIENT_SECRET_PATH} not found.\n"
            "Create OAuth2 Desktop App credentials in Google Cloud Console,\n"
            "enable the Photos Library API, and save the JSON here."
        )

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(
            str(TOKEN_PATH), _PHOTOS_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Detect credential type; web credentials need a fixed registered redirect URI
            import json as _json
            secret_data = _json.loads(CLIENT_SECRET_PATH.read_text())
            is_web = "web" in secret_data

            if is_web:
                print(
                    "   Web credential detected. Make sure http://localhost:8080 is\n"
                    "   listed under Authorized redirect URIs in Google Cloud Console."
                )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(CLIENT_SECRET_PATH), _PHOTOS_SCOPES,
                    redirect_uri="http://localhost:8080",
                )
                creds = flow.run_local_server(port=8080)
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(CLIENT_SECRET_PATH), _PHOTOS_SCOPES
                )
                creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())

    return creds


def _get_or_create_album(creds: Credentials) -> str:
    """
    Return the ID of the 'AI Restyle Output' Google Photos album.
    Creates it on first run and caches the ID in ALBUM_ID_CACHE.

    Note: the Google Photos Library API only allows writing to albums created
    by the same OAuth app — it cannot write to albums created via the UI.
    """
    if ALBUM_ID_CACHE.exists():
        album_id = ALBUM_ID_CACHE.read_text().strip()
        print(f"   Using cached album ID: {album_id}")
        return album_id

    auth = {"Authorization": f"Bearer {creds.token}"}
    resp = requests.post(
        "https://photoslibrary.googleapis.com/v1/albums",
        headers={**auth, "Content-Type": "application/json"},
        json={"album": {"title": "AI Restyle Output"}},
        timeout=15,
    )
    resp.raise_for_status()
    album_id = resp.json()["id"]
    product_url = resp.json().get("productUrl", "")
    ALBUM_ID_CACHE.write_text(album_id)
    print(f"   Created album 'AI Restyle Output': {product_url}")
    return album_id


def upload_to_photos_album(image_path: Path, album_id: str, creds: Credentials) -> str:
    """Upload image to Google Photos and add it to the album. Returns media item ID."""
    auth = {"Authorization": f"Bearer {creds.token}"}
    mime = "image/png" if image_path.suffix == ".png" else "image/jpeg"

    # Step 1: raw byte upload -> upload token
    with image_path.open("rb") as fh:
        up_resp = requests.post(
            "https://photoslibrary.googleapis.com/v1/uploads",
            headers={
                **auth,
                "Content-Type": "application/octet-stream",
                "X-Goog-Upload-Content-Type": mime,
                "X-Goog-Upload-Protocol": "raw",
                "X-Goog-Upload-File-Name": image_path.name,
            },
            data=fh.read(),
            timeout=120,
        )
    up_resp.raise_for_status()
    upload_token = up_resp.text.strip()

    # Step 2: create media item in the album
    create_resp = requests.post(
        "https://photoslibrary.googleapis.com/v1/mediaItems:batchCreate",
        headers={**auth, "Content-Type": "application/json"},
        json={
            "albumId": album_id,
            "newMediaItems": [{
                "simpleMediaItem": {
                    "fileName": image_path.name,
                    "uploadToken": upload_token,
                }
            }],
        },
        timeout=30,
    )
    create_resp.raise_for_status()

    results = create_resp.json().get("newMediaItemResults", [])
    if not results:
        raise RuntimeError("batchCreate returned no results.")

    status = results[0].get("status", {})
    msg = status.get("message", "")
    if msg and msg not in ("Success", "OK"):
        raise RuntimeError(f"Photos API error: {status}")

    return results[0].get("mediaItem", {}).get("id", "unknown")


# -- Entry point ---------------------------------------------------------------


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set the GEMINI_API_KEY environment variable first.")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # -- Authenticate with Google Photos once (cached in token.json) -----------
    print("-> Authenticating with Google Photos...")
    photos_creds = _auth_photos()

    print("-> Resolving Google Photos album...")
    album_id = _get_or_create_album(photos_creds)

    # -- iCloud -> Gemini (retry with a fresh photo if Gemini blocks) ----------
    for attempt in range(5):
        print("-> Fetching random photo from iCloud shared album...")
        raw_bytes, guid = download_random_photo(ICLOUD_ALBUM_TOKEN)
        print(f"   Photo {guid} downloaded ({len(raw_bytes):,} bytes)")

        print("-> Sending to Gemini for restyling...")
        try:
            result_bytes, ext = restyle_with_gemini(raw_bytes, api_key)
            break
        except RuntimeError as e:
            print(f"   {e} — fetching a different photo (attempt {attempt + 1}/5)...")
            if attempt == 4:
                raise
    else:
        raise RuntimeError("All 5 photos were blocked by Gemini.")

    uid = uuid.uuid4().hex[:8]
    out_path = OUTPUT_DIR / f"restyle_{uid}.{ext}"
    out_path.write_bytes(result_bytes)
    print(f"   Saved locally -> {out_path}")

    # -- Upload to Google Photos -----------------------------------------------
    print("-> Uploading to Google Photos album...")
    media_id = upload_to_photos_album(out_path, album_id, photos_creds)
    print(f"Done -> media item ID: {media_id}")


if __name__ == "__main__":
    main()
