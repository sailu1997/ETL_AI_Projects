import sqlite3
import os
import uuid
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import json
from contextlib import contextmanager

def make_json_serializable(obj):
    """Clean object for JSON serialization - handle NaN, infinity, and other non-serializable values"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (str, int, bool, type(None))):
        return obj
    else:
        return str(obj)

class DatabaseManager:
    def __init__(self, db_path: str = "user_sessions.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            ''')
            
            # Create files table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    uploaded_filename TEXT NOT NULL,
                    cleaned_filename TEXT,
                    file_type TEXT NOT NULL,
                    file_size INTEGER,
                    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_cleaned BOOLEAN DEFAULT FALSE,
                    metadata TEXT,
                    cleaning_log TEXT,
                    impact_metrics TEXT,
                    FOREIGN KEY (session_id) REFERENCES users (session_id)
                )
            ''')
            
            # Create user_analytics table for tracking usage
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    action_details TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES users (session_id)
                )
            ''')
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable row factory for dict-like access
        try:
            yield conn
        finally:
            conn.close()
    
    def create_user_session(self, session_id: str = None) -> str:
        """Create a new user session and return session_id"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (session_id, created_at, last_activity)
                VALUES (?, ?, ?)
            ''', (session_id, datetime.now(), datetime.now()))
            conn.commit()
        
        return session_id
    
    def get_user_session(self, session_id: str) -> Optional[Dict]:
        """Get user session information"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM users WHERE session_id = ? AND is_active = TRUE
            ''', (session_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def update_user_activity(self, session_id: str):
        """Update user's last activity timestamp"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET last_activity = ? WHERE session_id = ?
            ''', (datetime.now(), session_id))
            conn.commit()
    
    def deactivate_user_session(self, session_id: str):
        """Deactivate a user session"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET is_active = FALSE WHERE session_id = ?
            ''', (session_id,))
            conn.commit()
    
    def add_file_record(self, session_id: str, original_filename: str, 
                       uploaded_filename: str, file_type: str, 
                       file_size: int, metadata: Dict = None) -> str:
        """Add a file record to the database"""
        file_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO files (file_id, session_id, original_filename, uploaded_filename, 
                                 file_type, file_size, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (file_id, session_id, original_filename, uploaded_filename, 
                  file_type, file_size, json.dumps(metadata) if metadata else None))
            conn.commit()
        
        return file_id
    
    def get_user_files(self, session_id: str) -> List[Dict]:
        """Get all files for a user session"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM files WHERE session_id = ? ORDER BY upload_timestamp DESC
            ''', (session_id,))
            rows = cursor.fetchall()
            
            files = []
            for row in rows:
                file_data = dict(row)
                # Parse JSON fields
                if file_data.get('metadata'):
                    file_data['metadata'] = json.loads(file_data['metadata'])
                    file_data['metadata'] = make_json_serializable(file_data['metadata'])
                if file_data.get('cleaning_log'):
                    file_data['cleaning_log'] = json.loads(file_data['cleaning_log'])
                    file_data['cleaning_log'] = make_json_serializable(file_data['cleaning_log'])
                if file_data.get('impact_metrics'):
                    file_data['impact_metrics'] = json.loads(file_data['impact_metrics'])
                    file_data['impact_metrics'] = make_json_serializable(file_data['impact_metrics'])
                files.append(file_data)
            
            return files
    
    def get_file_by_id(self, file_id: str, session_id: str = None) -> Optional[Dict]:
        """Get a specific file by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if session_id:
                cursor.execute('''
                    SELECT * FROM files WHERE file_id = ? AND session_id = ?
                ''', (file_id, session_id))
            else:
                cursor.execute('''
                    SELECT * FROM files WHERE file_id = ?
                ''', (file_id,))
            
            row = cursor.fetchone()
            if row:
                file_data = dict(row)
                # Parse JSON fields
                if file_data.get('metadata'):
                    file_data['metadata'] = json.loads(file_data['metadata'])
                    file_data['metadata'] = make_json_serializable(file_data['metadata'])
                if file_data.get('cleaning_log'):
                    file_data['cleaning_log'] = json.loads(file_data['cleaning_log'])
                    file_data['cleaning_log'] = make_json_serializable(file_data['cleaning_log'])
                if file_data.get('impact_metrics'):
                    file_data['impact_metrics'] = json.loads(file_data['impact_metrics'])
                    file_data['impact_metrics'] = make_json_serializable(file_data['impact_metrics'])
                return file_data
            return None
    
    def update_file_cleaned_status(self, file_id: str, cleaned_filename: str, 
                                  cleaning_log: List[str] = None, 
                                  impact_metrics: Dict = None):
        """Update file's cleaned status"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE files SET is_cleaned = TRUE, cleaned_filename = ?, 
                               cleaning_log = ?, impact_metrics = ?
                WHERE file_id = ?
            ''', (cleaned_filename, 
                  json.dumps(cleaning_log) if cleaning_log else None,
                  json.dumps(impact_metrics) if impact_metrics else None,
                  file_id))
            conn.commit()
    
    def delete_file_record(self, file_id: str, session_id: str):
        """Delete a file record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM files WHERE file_id = ? AND session_id = ?
            ''', (file_id, session_id))
            conn.commit()
    
    def log_user_action(self, session_id: str, action_type: str, action_details: Dict = None):
        """Log a user action for analytics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_analytics (session_id, action_type, action_details)
                VALUES (?, ?, ?)
            ''', (session_id, action_type, 
                  json.dumps(action_details) if action_details else None))
            conn.commit()
    
    def get_user_analytics(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get user analytics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM user_analytics 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (session_id, limit))
            rows = cursor.fetchall()
            
            analytics = []
            for row in rows:
                analytics_data = dict(row)
                if analytics_data.get('action_details'):
                    analytics_data['action_details'] = json.loads(analytics_data['action_details'])
                analytics.append(analytics_data)
            
            return analytics
    
    def cleanup_inactive_sessions(self, days_threshold: int = 7):
        """Clean up inactive sessions older than threshold"""
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get inactive sessions
            cursor.execute('''
                SELECT session_id FROM users 
                WHERE last_activity < ? AND is_active = TRUE
            ''', (cutoff_date,))
            inactive_sessions = [row['session_id'] for row in cursor.fetchall()]
            
            # Deactivate sessions
            for session_id in inactive_sessions:
                cursor.execute('''
                    UPDATE users SET is_active = FALSE WHERE session_id = ?
                ''', (session_id,))
            
            conn.commit()
            return len(inactive_sessions)
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get comprehensive statistics for a user session - matches backend exactly"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get user info
            cursor.execute('SELECT * FROM users WHERE session_id = ?', (session_id,))
            user_row = cursor.fetchone()
            
            if not user_row:
                return None
            
            # Get file count
            cursor.execute('''
                SELECT COUNT(*) as total_files, 
                       COUNT(CASE WHEN is_cleaned = TRUE THEN 1 END) as cleaned_files
                FROM files WHERE session_id = ?
            ''', (session_id,))
            file_stats = cursor.fetchone()
            
            # Get action count
            cursor.execute('''
                SELECT COUNT(*) as total_actions,
                       COUNT(CASE WHEN action_type = 'upload' THEN 1 END) as uploads,
                       COUNT(CASE WHEN action_type = 'clean' THEN 1 END) as cleans,
                       COUNT(CASE WHEN action_type = 'chat' THEN 1 END) as chats
                FROM user_analytics WHERE session_id = ?
            ''', (session_id,))
            action_stats = cursor.fetchone()
            
            return {
                'session_id': session_id,
                'created_at': user_row['created_at'],
                'last_activity': user_row['last_activity'],
                'is_active': user_row['is_active'],
                'total_files': file_stats['total_files'] if file_stats else 0,
                'cleaned_files': file_stats['cleaned_files'] if file_stats else 0,
                'total_actions': action_stats['total_actions'] if action_stats else 0,
                'uploads': action_stats['uploads'] if action_stats else 0,
                'cleans': action_stats['cleans'] if action_stats else 0,
                'chats': action_stats['chats'] if action_stats else 0
            }
    
    def update_conversation_history(self, session_id: str, conversation_history: List[Dict[str, Any]]):
        """Update conversation history for a session"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Store conversation history as JSON in user_analytics
            cursor.execute('''
                INSERT OR REPLACE INTO user_analytics (session_id, action_type, action_details, timestamp)
                VALUES (?, 'conversation_history', ?, ?)
            ''', (session_id, json.dumps(conversation_history), datetime.now().isoformat()))
            
            conn.commit()
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT action_details FROM user_analytics 
                WHERE session_id = ? AND action_type = 'conversation_history'
                ORDER BY timestamp DESC 
                LIMIT 1
            ''', (session_id,))
            
            row = cursor.fetchone()
            if row and row['action_details']:
                try:
                    return json.loads(row['action_details'])
                except json.JSONDecodeError:
                    return []
            
            return []
    
    def update_conversation_memory(self, session_id: str, memory_data: Dict[str, Any]) -> bool:
        """Update conversation memory for a session"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create conversation_memory table if it doesn't exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversation_memory (
                        session_id TEXT PRIMARY KEY,
                        memory_data TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES users (session_id)
                    )
                ''')
                
                # Insert or update conversation memory
                cursor.execute('''
                    INSERT OR REPLACE INTO conversation_memory (session_id, memory_data, updated_at)
                    VALUES (?, ?, ?)
                ''', (session_id, json.dumps(memory_data), datetime.now()))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error updating conversation memory: {e}")
            return False
    
    def get_conversation_memory(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation memory for a session"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create conversation_memory table if it doesn't exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversation_memory (
                        session_id TEXT PRIMARY KEY,
                        memory_data TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES users (session_id)
                    )
                ''')
                
                # Get conversation memory
                cursor.execute('''
                    SELECT memory_data FROM conversation_memory WHERE session_id = ?
                ''', (session_id,))
                
                row = cursor.fetchone()
                if row:
                    return json.loads(row['memory_data'])
                return None
                
        except Exception as e:
            print(f"Error getting conversation memory: {e}")
            return None
    
    def update_file_metadata(self, file_id: str, metadata: Dict[str, Any]) -> bool:
        """Update file metadata in the database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Update the metadata field for the file
                cursor.execute('''
                    UPDATE files 
                    SET metadata = ?, 
                        upload_timestamp = CURRENT_TIMESTAMP
                    WHERE file_id = ?
                ''', (json.dumps(metadata), file_id))
                
                conn.commit()
                print(f"DEBUG: Updated metadata for file {file_id}")
                return True
                
        except Exception as e:
            print(f"Error updating file metadata: {e}")
            return False

# Global database manager instance
db_manager = DatabaseManager() 