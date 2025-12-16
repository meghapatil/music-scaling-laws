"""
Memory-safe MIDI converter with checkpointing.
"""

from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings('ignore')

def convert_one_midi(args):
    """Convert one MIDI file."""
    midi_path, output_dir = args
    output_path = output_dir / f"{midi_path.stem}.abc"
    if output_path.exists() and output_path.stat().st_size > 50:
        return (True, "skipped")
    
    try:
        import music21
        print(f"DEBUG: Attempting to parse {midi_path.name}")
        midi = music21.converter.parse(str(midi_path), forceSource=True, quantizePost=False)
        notes = list(midi.flatten().notesAndRests)[:200]
        
        if len(notes) < 10:
            return (False, "too_few")
        
        abc_notes = []
        for elem in notes:
            if isinstance(elem, music21.note.Note):
                name = elem.pitch.name
                octave = elem.pitch.octave
                abc_notes.append(name.lower() if octave >= 5 else name.upper())
            elif isinstance(elem, music21.note.Rest):
                abc_notes.append("z")
        
        parts = ["X:1", f"T:{midi_path.stem[:40]}", "M:4/4", "L:1/8", "K:C"]
        for i in range(0, len(abc_notes), 16):
            parts.append(" ".join(abc_notes[i:i+16]) + " |")
        
        abc_string = "\n".join(parts)
        
        with open(output_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(abc_string)
        
        return (True, "success")
        
    except Exception as e:
        return (False, str(e)[:30])


def safe_parallel_convert(midi_dir, output_dir, batch_size=5000, num_processes=4):
    """
    Safe parallel conversion with batching and fewer processes.
    Uses only 4 processes to avoid memory issues.
    """
    midi_dir = Path(midi_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Scanning for MIDI files...")
    midi_files = list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi"))
    
    existing = len(list(output_dir.glob("*.abc")))
    print(f"Found {len(midi_files)} MIDI files")
    print(f"Already converted: {existing} files")
    print(f"Using {num_processes} processes (memory-safe)")
    print(f"Processing in batches of {batch_size}")
    
    total_successful = existing
    
    # Process in batches to avoid memory issues
    for batch_num in range(0, len(midi_files), batch_size):
        batch_files = midi_files[batch_num:batch_num + batch_size]
        
        print(f"\nBatch {batch_num//batch_size + 1}: "
              f"Processing files {batch_num} to {min(batch_num + batch_size, len(midi_files))}")
        
        args_list = [(f, output_dir) for f in batch_files]
        
        try:
            with Pool(processes=num_processes) as pool:
                results = list(tqdm(
                    pool.imap(convert_one_midi, args_list, chunksize=50),
                    total=len(batch_files),
                    desc=f"Batch {batch_num//batch_size + 1}",
                    unit="files"
                ))
            
            batch_success = sum(1 for success, _ in results if success)
            total_successful += batch_success
            
            print(f"Batch complete. Total converted so far: {total_successful}")
            
            # Small delay between batches
            time.sleep(2)
            
        except Exception as e:
            print(f"Error in batch: {e}")
            print("Continuing to next batch...")
            continue
    
    print(f"\nConversion complete! Total files: {total_successful}")
    return total_successful


if __name__ == "__main__":
    safe_parallel_convert(
        midi_dir="data/raw_midi",
        output_dir="data/abc_notation",
        batch_size=5000,
        num_processes=4  # Conservative - less memory usage
    )