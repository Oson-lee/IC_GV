# File path: GiantVirus_Classification/src/sequence_processor.py

from Bio import SeqIO
import numpy as np

def pad_sequence(sequence: str, target_length: int, pad_char: str = 'N') -> str:
    """
    Pad the sequence with a specified character (usually 'N') at the end 
    if its length is less than the target length.
    """
    if len(sequence) >= target_length:
        return sequence
    padding_length = target_length - len(sequence)
    return sequence + (pad_char * padding_length)

def sliding_window_slice(sequence: str, window_size: int, stride: int) -> list:
    """
    Perform sliding window slicing on a long sequence.
    """
    seq_length = len(sequence)
    slices = []
    
    # If the sequence is smaller than the window, pad it directly and return
    if seq_length < window_size:
        return [pad_sequence(sequence, window_size)]
        
    for start in range(0, seq_length - window_size + 1, stride):
        end = start + window_size
        slices.append(sequence[start:end])
        
    # Handle the tail part that is shorter than a full window
    if seq_length > window_size and (seq_length - window_size) % stride != 0:
        slices.append(sequence[-window_size:])
        
    # Remove duplicates
    slices = list(dict.fromkeys(slices))
    return slices

def process_fasta_file(fasta_path: str, window_size: int = 10000, stride: int = 5000) -> list:
    """
    Read a single FASTA file and return a list of sliced sequence strings.
    """
    all_slices = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_str = str(record.seq).upper()  
        
        slices = sliding_window_slice(seq_str, window_size, stride)
        all_slices.extend(slices)
        
    return all_slices