use std::collections::HashMap;
use std::sync::Arc;

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use rusqlite::Connection;

use super::config::ensure_parent_dir;

#[derive(Clone)]
pub struct SharedConnection {
    pub conn: Arc<Mutex<Connection>>,
    pub path: String,
}

struct Entry {
    conn: Arc<Mutex<Connection>>,
    refs: usize,
}

static CONNECTIONS: Lazy<Mutex<HashMap<String, Entry>>> = Lazy::new(|| Mutex::new(HashMap::new()));

fn is_connection_healthy(conn: &Arc<Mutex<Connection>>) -> bool {
    conn.lock()
        .query_row("SELECT 1", [], |_row| Ok::<i32, rusqlite::Error>(1))
        .is_ok()
}

fn open_connection(path: &str) -> anyhow::Result<Arc<Mutex<Connection>>> {
    ensure_parent_dir(path)?;
    let conn = Connection::open(path)?;
    conn.execute_batch(
        r#"
        PRAGMA journal_mode = WAL;
        PRAGMA foreign_keys = ON;
        "#,
    )?;
    Ok(Arc::new(Mutex::new(conn)))
}

pub fn get_lcm_connection(db_path: &str) -> anyhow::Result<SharedConnection> {
    let path = db_path.trim().to_string();
    if path.is_empty() {
        anyhow::bail!("database path is required");
    }

    let mut map = CONNECTIONS.lock();
    if let Some(entry) = map.get_mut(&path) {
        if is_connection_healthy(&entry.conn) {
            entry.refs += 1;
            return Ok(SharedConnection {
                conn: entry.conn.clone(),
                path,
            });
        }
        map.remove(&path);
    }

    let conn = open_connection(&path)?;
    map.insert(
        path.clone(),
        Entry {
            conn: conn.clone(),
            refs: 1,
        },
    );
    Ok(SharedConnection { conn, path })
}

pub fn close_lcm_connection(db_path: Option<&str>) {
    let mut map = CONNECTIONS.lock();
    if let Some(path) = db_path.map(str::trim).filter(|p| !p.is_empty()) {
        if let Some(entry) = map.get_mut(path) {
            if entry.refs > 0 {
                entry.refs -= 1;
            }
            if entry.refs == 0 {
                map.remove(path);
            }
        }
        return;
    }
    map.clear();
}
