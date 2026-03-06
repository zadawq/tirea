use std::path::Path;
use tokio::io::AsyncWriteExt;

/// Validate an ID for filesystem safety.
///
/// Rejects empty strings, path separators, `..`, null bytes, and control characters.
pub fn validate_fs_id(id: &str, label: &str) -> Result<(), String> {
    if id.trim().is_empty() {
        return Err(format!("{label} cannot be empty"));
    }
    if id.contains('/')
        || id.contains('\\')
        || id.contains("..")
        || id.contains('\0')
        || id.chars().any(|c| c.is_control())
    {
        return Err(format!("{label} contains invalid characters: {id:?}"));
    }
    Ok(())
}

/// Atomic JSON write: write to tmp file, fsync, then rename into place.
///
/// If the target already exists and rename fails with `AlreadyExists`,
/// falls back to remove-then-rename (for non-POSIX platforms).
pub async fn atomic_json_write(
    base_dir: &Path,
    filename: &str,
    content: &str,
) -> Result<(), std::io::Error> {
    if !base_dir.exists() {
        tokio::fs::create_dir_all(base_dir).await?;
    }

    let target = base_dir.join(filename);
    let tmp_path = base_dir.join(format!(
        ".{}.{}.tmp",
        filename.trim_end_matches(".json"),
        uuid::Uuid::new_v4().simple()
    ));

    let write_result = async {
        let mut file = tokio::fs::File::create(&tmp_path).await?;
        file.write_all(content.as_bytes()).await?;
        file.flush().await?;
        file.sync_all().await?;
        drop(file);
        match tokio::fs::rename(&tmp_path, &target).await {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                tokio::fs::remove_file(&target).await?;
                tokio::fs::rename(&tmp_path, &target).await?;
            }
            Err(e) => return Err(e),
        }
        Ok::<(), std::io::Error>(())
    }
    .await;

    if let Err(e) = write_result {
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return Err(e);
    }
    Ok(())
}

/// Scan a directory for `.json` files and return their file stems.
pub async fn scan_json_stems(path: &Path) -> Result<Vec<String>, std::io::Error> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut entries = tokio::fs::read_dir(path).await?;
    let mut stems = Vec::new();
    while let Some(entry) = entries.next_entry().await? {
        let p = entry.path();
        if p.extension().is_some_and(|ext| ext == "json") {
            if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                stems.push(stem.to_string());
            }
        }
    }
    Ok(stems)
}
