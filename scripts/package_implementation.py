import os
import shutil
import zipfile
from pathlib import Path

def copy_clean_tree(src_dir, dest_dir, ignore_names=None):
    if ignore_names is None:
        ignore_names = {
            ".venv", "__pycache__", ".pytest_cache", ".git", "data.zip", 
            "submission", ".system_generated", "dist", "build", "*.pyc"
        }
    
    src_path = Path(src_dir).resolve()
    dest_path = Path(dest_dir).resolve()
    
    if dest_path.exists():
        shutil.rmtree(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Items to copy from project root
    dirs_to_copy = ["src", "tests", "configs", "scripts"]
    files_to_copy = ["pyproject.toml", "README.md", ".python-version", ".gitignore"]
    
    for d in dirs_to_copy:
        s_d = src_path / d
        if s_d.exists():
            d_d = dest_path / d
            print(f"Copying directory: {d} -> {d_d}")
            shutil.copytree(
                s_d, d_d,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.log")
            )
            
    for f in files_to_copy:
        s_f = src_path / f
        if s_f.exists():
            d_f = dest_path / f
            print(f"Copying file: {f} -> {d_f}")
            shutil.copy2(s_f, d_f)

def create_zip(source_dir, output_zip):
    src = Path(source_dir).resolve()
    out = Path(output_zip).resolve()
    
    print(f"Creating ZIP archive: {out} from {src}")
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(src):
            for file in files:
                file_path = Path(root) / file
                archive_name = file_path.relative_to(src.parent)
                zipf.write(file_path, archive_name)
                
    print(f"Successfully generated ZIP: {out} ({out.stat().st_size} bytes)")

if __name__ == "__main__":
    impl_dir = "submission/implementation"
    zip_path = "submission/LLM-FINANCE-Implementation.zip"
    
    copy_clean_tree(".", impl_dir)
    create_zip(impl_dir, zip_path)
