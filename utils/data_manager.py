import os
import sqlite3
import pickle



class SqliteDict:
    """Dictionary-like interface for SQLite storage with incremental updates."""
    
    def __init__(self, filename):
        self.conn = sqlite3.connect(filename, check_same_thread=False)
        self.conn.execute('CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value BLOB)')
        self.conn.commit()
    
    def __getitem__(self, key):
        result = self.conn.execute('SELECT value FROM cache WHERE key = ?', (key,)).fetchone()
        if result is None:
            raise KeyError(key)
        return pickle.loads(result[0])
    
    def __setitem__(self, key, value):
        self.conn.execute('INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)', 
                         (key, pickle.dumps(value)))
        self.conn.commit()
    
    def __contains__(self, key):
        return self.conn.execute('SELECT 1 FROM cache WHERE key = ?', (key,)).fetchone() is not None
    
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
    
    def keys(self):
        return [row[0] for row in self.conn.execute('SELECT key FROM cache')]
    
    def items(self):
        for key in self.keys():
            yield key, self[key]
    
    def close(self):
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()