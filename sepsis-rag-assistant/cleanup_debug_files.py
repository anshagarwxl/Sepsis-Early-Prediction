#!/usr/bin/env python3
"""
Safe cleanup script to remove debug and test files from the project.
Keeps all main RAG pipeline files intact.
"""

import os
import glob
from pathlib import Path

def find_debug_files():
    """Find all debug/test files matching specified patterns."""
    patterns = [
        "*_test.py",
        "debug_*.py", 
        "temp_*.py",
        "api_key_check.py",
        "test_*.py",
        "*_debug.py",
        "verify_*.py",
        "ingest_documents.py",  # Your testing script
        "test_api_key.py",
        "test_gemini_api.py"
    ]
    
    # Protected files - NEVER delete these
    protected_files = {
        "app.py",
        "rag_system.py", 
        "data_prep.py",
        "scoring.py",
        "ml_training.py",
        "generate_data.py",
        "cleanup_debug_files.py",  # Don't delete self
        "run_app.ps1",  # Protect launch scripts
        "run_app.bat"   # Protect launch scripts
    }
    
    # Protected directories - NEVER scan these
    protected_dirs = {
        "data", ".venv", "models", "config", "__pycache__", ".git"
    }
    
    found_files = []
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        for file in matches:
            filename = os.path.basename(file)
            file_dir = os.path.dirname(file)
            
            # Skip if in protected directory
            if any(pdir in file_dir for pdir in protected_dirs):
                continue
                
            if filename not in protected_files and os.path.isfile(file):
                found_files.append(file)
    
    return sorted(list(set(found_files)))  # Remove duplicates and sort

def main():
    print("🧹 Debug File Cleanup Script")
    print("=" * 40)
    
    # Find files to delete
    files_to_delete = find_debug_files()
    
    if not files_to_delete:
        print("✅ No debug/test files found to clean up!")
        return
    
    print(f"📋 Found {len(files_to_delete)} debug/test files:")
    print()
    
    for i, file in enumerate(files_to_delete, 1):
        file_size = os.path.getsize(file) if os.path.exists(file) else 0
        print(f"  {i:2d}. {file} ({file_size} bytes)")
    
    print()
    print("🔒 Protected files (will NOT be deleted):")
    protected = ["app.py", "rag_system.py", "data_prep.py", "scoring.py", "ml_training.py"]
    for pfile in protected:
        if os.path.exists(pfile):
            print(f"     ✓ {pfile}")
    
    print()
    
    # Ask for confirmation
    while True:
        response = input("❓ Delete these debug files? (Y/N): ").strip().upper()
        if response in ['Y', 'YES']:
            break
        elif response in ['N', 'NO']:
            print("❌ Cleanup cancelled.")
            return
        else:
            print("Please enter Y or N")
    
    # Delete files
    deleted_count = 0
    failed_count = 0
    
    print("\n🗑️  Deleting files...")
    
    for file in files_to_delete:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"   ✓ Deleted: {file}")
                deleted_count += 1
            else:
                print(f"   ⚠️ Not found: {file}")
        except Exception as e:
            print(f"   ❌ Failed to delete {file}: {e}")
            failed_count += 1
    
    print()
    print("📊 Cleanup Summary:")
    print(f"   ✅ Deleted: {deleted_count} files")
    if failed_count > 0:
        print(f"   ❌ Failed: {failed_count} files")
    print(f"   🔒 Protected: All main pipeline files kept safe")
    
    print("\n🎉 Cleanup complete!")
    print("   Your RAG system is unchanged and will continue working perfectly.")

if __name__ == "__main__":
    main()