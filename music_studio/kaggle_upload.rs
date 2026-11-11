//! Resumable original WAV/MP3 imports for the Kaggle/Cloudflare browser build.
use axum::{body::Bytes, extract::{Path, State}, http::StatusCode, Json};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, io::{Read, Seek, SeekFrom, Write}, path::PathBuf, sync::{Mutex, OnceLock}};
use crate::{AppState, ApiError, api_error};
type Answer = Result<Json<Value>, (StatusCode, Json<ApiError>)>;

#[derive(Deserialize)]
pub struct Start { filename: String, bytes: u64, fingerprint: String }
#[derive(Serialize, Deserialize)]
struct Receipt { filename: String, bytes: u64, fingerprint: String, received: u64, chunks: Vec<Chunk>, song_id: Option<String> }
#[derive(Serialize, Deserialize)]
struct Chunk { offset: u64, bytes: u64, sha256: String }
#[derive(Deserialize)]
pub struct Position { offset: u64, sha256: String }
fn fail(error: impl ToString) -> (StatusCode, Json<ApiError>) { api_error(StatusCode::BAD_REQUEST, error.to_string()) }
fn folder() -> PathBuf { std::env::var_os("YUE_LIBRARY_ROOT").map(PathBuf::from).unwrap_or_else(std::env::temp_dir).join("uploads") }
fn paths(id: &str) -> Result<(PathBuf, PathBuf), (StatusCode, Json<ApiError>)> {
    if id.len() != 32 || !id.bytes().all(|b| b.is_ascii_hexdigit()) { return Err(fail("Invalid upload ID")) }
    Ok((folder().join(format!("{id}.json")), folder().join(format!("{id}.part"))))
}
fn gate() -> &'static Mutex<()> { static LOCK: OnceLock<Mutex<()>> = OnceLock::new(); LOCK.get_or_init(|| Mutex::new(())) }
fn read(path: &PathBuf) -> Result<Receipt, (StatusCode, Json<ApiError>)> { serde_json::from_slice(&fs::read(path).map_err(fail)?).map_err(fail) }
fn save(path: &PathBuf, receipt: &Receipt) -> Result<(), (StatusCode, Json<ApiError>)> {
    let temp = path.with_extension("new"); fs::write(&temp, serde_json::to_vec(receipt).map_err(fail)?).map_err(fail)?;
    fs::rename(temp, path).map_err(fail)
}
pub async fn start(Json(input): Json<Start>) -> Answer {
    let _lock = gate().lock().map_err(fail)?;
    let name = std::path::Path::new(&input.filename).file_name().and_then(|x| x.to_str()).unwrap_or("").to_string();
    let ext = std::path::Path::new(&name).extension().and_then(|x| x.to_str()).unwrap_or("").to_lowercase();
    if name.is_empty() || !matches!(ext.as_str(), "wav" | "mp3") || input.bytes == 0 || input.bytes > 2 * 1024 * 1024 * 1024 { return Err(fail("Choose an original WAV or MP3, at most 2 GiB")) }
    if input.fingerprint.len() != 64 || !input.fingerprint.bytes().all(|b| b.is_ascii_hexdigit()) { return Err(fail("Invalid upload fingerprint")) }
    fs::create_dir_all(folder()).map_err(fail)?;
    for entry in fs::read_dir(folder()).map_err(fail)?.flatten() {
        if entry.path().extension().and_then(|x| x.to_str()) == Some("json") {
            if let Ok(old) = read(&entry.path()) {
                if old.filename == name && old.bytes == input.bytes && old.fingerprint == input.fingerprint {
                    let id = entry.path().file_stem().unwrap().to_string_lossy().into_owned();
                    return Ok(Json(json!({"id":id, "received":old.received,"chunks":old.chunks,"song_id":old.song_id})))
                }
            }
        }
    }
    let id = uuid::Uuid::now_v7().simple().to_string(); let (meta, part) = paths(&id)?;
    let receipt = Receipt { filename:name, bytes:input.bytes, fingerprint:input.fingerprint, received:0, chunks:vec![], song_id:None };
    fs::OpenOptions::new().write(true).create_new(true).open(part).map_err(fail)?; save(&meta, &receipt)?;
    Ok(Json(json!({"id":id,"received":0,"chunks":[]})))
}
pub async fn status(Path(id): Path<String>) -> Answer {
    let (meta, _) = paths(&id)?; let receipt = read(&meta)?;
    Ok(Json(json!({"id":id,"received":receipt.received,"chunks":receipt.chunks,"song_id":receipt.song_id})))
}
pub async fn chunk(Path(id): Path<String>, axum::extract::Query(position): axum::extract::Query<Position>, data: Bytes) -> Answer {
    let _lock = gate().lock().map_err(fail)?; let (meta, part) = paths(&id)?; let mut receipt = read(&meta)?;
    if receipt.song_id.is_some() { return Err(fail("Upload already imported")) }
    if data.is_empty() || data.len() > 4 * 1024 * 1024 || position.offset + data.len() as u64 > receipt.bytes { return Err(fail("Invalid upload chunk size")) }
    let digest = Sha256::digest(&data).iter().map(|b| format!("{b:02x}")).collect::<String>();
    if digest != position.sha256 { return Err(fail("Chunk checksum failed")) }
    if position.offset < receipt.received {
        if receipt.chunks.iter().any(|c| c.offset == position.offset && c.bytes == data.len() as u64 && c.sha256 == digest) { return Ok(Json(json!({"received":receipt.received}))) }
        return Err(fail("Previously saved chunk differs"))
    }
    if position.offset != receipt.received { return Err(fail("Chunk offset differs; resume from status")) }
    let mut file = fs::OpenOptions::new().write(true).read(true).open(&part).map_err(fail)?;
    // Discard only an unacknowledged tail after interruption of the metadata write.
    if file.metadata().map_err(fail)?.len() < receipt.received { return Err(fail("Saved upload is truncated")) }
    file.set_len(receipt.received).map_err(fail)?; file.seek(SeekFrom::Start(receipt.received)).map_err(fail)?;
    file.write_all(&data).map_err(fail)?; file.sync_data().map_err(fail)?;
    receipt.chunks.push(Chunk { offset:position.offset, bytes:data.len() as u64, sha256:digest }); receipt.received += data.len() as u64;
    save(&meta, &receipt)?; Ok(Json(json!({"received":receipt.received})))
}
pub async fn finish(State(state): State<AppState>, Path(id): Path<String>) -> Answer {
    let _lock = gate().lock().map_err(fail)?; let (meta, part) = paths(&id)?; let mut receipt = read(&meta)?;
    if let Some(old) = state.library.list_songs().map_err(fail)?.into_iter().find(|s| s.metadata.get("kaggle_upload_id").and_then(Value::as_str) == Some(&id)) {
        receipt.song_id = Some(old.id.clone()); save(&meta, &receipt)?;
        if part.exists() { fs::remove_file(part).map_err(fail)?; }
        return Ok(Json(json!({"song_id":old.id,"ready":true})))
    }
    if receipt.received != receipt.bytes { return Err(fail("Upload incomplete")) }
    let mut audio = Vec::new(); fs::File::open(&part).map_err(fail)?.read_to_end(&mut audio).map_err(fail)?;
    if audio.len() as u64 != receipt.bytes { return Err(fail("Upload length verification failed")) }
    let ext = std::path::Path::new(&receipt.filename).extension().and_then(|x| x.to_str()).unwrap_or("").to_lowercase();
    let duration = crate::library::audio_duration_seconds(&audio, &ext, None);
    let digest = Sha256::digest(&audio).iter().map(|b| format!("{b:02x}")).collect::<String>();
    let title = std::path::Path::new(&receipt.filename).file_stem().unwrap().to_string_lossy().into_owned();
    let song = state.library.import_audio_song(crate::library::AudioImportInput {
        title, caption:String::new(), lyrics:String::new(), metadata:json!({"imported_filename":receipt.filename,"duration_seconds":duration,"source_sha256":digest,"source_bytes":receipt.bytes,"kaggle_upload_id":id}),
        generation_settings:Value::Null, engine_id:"imported-audio".into(), profile_id:None, source:"resumable_audio_import".into(), audio_extension:ext, audio
    }).map_err(fail)?.song;
    receipt.song_id = Some(song.id.clone()); save(&meta, &receipt)?; fs::remove_file(part).map_err(fail)?;
    Ok(Json(json!({"song_id":song.id,"ready":true,"source_sha256":digest})))
}
