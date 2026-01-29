use memvid_core::{Memvid, MemvidError};
use std::path::{Path, PathBuf};
use tempfile::TempDir;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn create_tmp_memvid() -> Result<(TempDir, PathBuf), Box<dyn std::error::Error>> {
    let dir = TempDir::new()?;
    let path = dir.path().join("test.mv2");
    // Create new memory
    {
        let mut mem = Memvid::create(&path)?;
        mem.commit()?;
    }
    Ok((dir, path))
}

fn open_tmp_memvid(path: &Path) -> Result<Memvid, Box<dyn std::error::Error>> {
    Ok(Memvid::open(path)?)
}

#[test]
fn test_vec_model_consistency() -> TestResult {
    let (_dir, path) = create_tmp_memvid()?;

    // 1. Create index and set model "A"
    {
        let mut memvid = open_tmp_memvid(&path)?;
        memvid.enable_vec()?;
        memvid.set_vec_model("model-a")?;
        memvid.commit()?;
    }

    // 2. Open and verify "model-a" matches and "model-b" fails
    {
        let mut memvid = open_tmp_memvid(&path)?;
        // Should succeed (idempotent)
        memvid.set_vec_model("model-a")?;

        // Should fail
        let result = memvid.set_vec_model("model-b");
        assert!(result.is_err());
        match result {
            Err(MemvidError::ModelMismatch { expected, actual }) => {
                assert_eq!(expected, "model-a");
                assert_eq!(actual, "model-b");
            }
            _ => panic!("Expected ModelMismatch error"),
        }
    }

    Ok(())
}

#[test]
fn test_vec_model_persistence() -> TestResult {
    let (_dir, path) = create_tmp_memvid()?;

    // 1. Create index with model
    {
        let mut memvid = open_tmp_memvid(&path)?;
        memvid.enable_vec()?;
        memvid.set_vec_model("persistent-model")?;
        memvid.commit()?;
    }

    // 2. Open and check if model is loaded automatically
    {
        let mut memvid = open_tmp_memvid(&path)?;
        // We can inspect internal state via debug or by trying to set a mismatch
        let result = memvid.set_vec_model("wrong-model");
        assert!(result.is_err());

        // Verify set_vec_model("persistent-model") works (confirming loaded state)
        memvid.set_vec_model("persistent-model")?;
    }

    Ok(())
}
