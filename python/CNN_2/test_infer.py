import re
import numpy as np
import os

#finds a C-array by name and extracts its integers.
def extract_c_array(filepath, array_name):
    
    with open(filepath, 'r') as f:
        content = f.read()

    start_idx = content.find(array_name)
    if start_idx == -1:
        return None

    #extract everything between the first '{' and the matching ';'
    start_bracket = content.find('{', start_idx)
    end_bracket = content.find(';', start_bracket)
    block = content[start_bracket:end_bracket]

    #find all integers (including negative signs)
    nums = re.findall(r'-?\d+', block)
    return np.array([int(n) for n in nums], dtype=np.int32)

def get_test_image_names(filepath):
    """Dynamically finds all test image array names in the header."""
    with open(filepath, 'r') as f:
        content = f.read()
    return re.findall(r'int8_t (Person_\d+_Test_\d+)\[32\]\[32\]', content)

def hardware_inference(img_int8, W_conv_int8, W_linear_int8):
    """Simulates the Nios II + CIMmetry-1616 exact clock cycles."""
    
    # 1. Image Reshape (Extract 64 patches of 4x4)
    patches = np.zeros((64, 16), dtype=np.int32)
    idx = 0
    for r in range(0, 32, 4):
        for c in range(0, 32, 4):
            patches[idx] = img_int8[r:r+4, c:c+4].flatten()
            idx += 1

    # 2. Cycle 1: Conv -> ReLU -> Shift >> 6 -> Clamp
    acc1 = np.dot(patches, W_conv_int8) # Matrix Multiply
    acc1 = np.maximum(0, acc1)          # ReLU
    acc1_shifted = acc1 >> 6            # Hardware Scaling Division
    conv_out = np.clip(acc1_shifted, -128, 127).astype(np.int32)

    # 3. Nios II Quadrant Pooling (Integer Division by 16)
    grid = conv_out.reshape((8, 8, 16))
    q_TL = np.sum(grid[0:4, 0:4, :], axis=(0,1)) // 16
    q_TR = np.sum(grid[0:4, 4:8, :], axis=(0,1)) // 16
    q_BL = np.sum(grid[4:8, 0:4, :], axis=(0,1)) // 16
    q_BR = np.sum(grid[4:8, 4:8, :], axis=(0,1)) // 16
    matrix_A_cycle2 = np.stack([q_TL, q_TR, q_BL, q_BR])

    # 4. Cycle 2: Linear -> Shift >> 6 -> Clamp
    acc2 = np.dot(matrix_A_cycle2, W_linear_int8)
    acc2_shifted = np.right_shift(acc2, 6) # Handles negative two's complement perfectly
    final_sig = np.clip(acc2_shifted, -128, 127).flatten()
    
    return final_sig

#distance calculator
def calc_pytorch_distance(sig_int8, anchor_int8):
    """Converts the 8-bit integers back to floats to match the 1.7 PyTorch threshold"""
    float_sig = sig_int8 / 64.0
    float_anchor = anchor_int8 / 64.0
    # Euclidean L2 Distance
    return np.sqrt(np.sum((float_sig - float_anchor)**2))


if __name__ == "__main__":
    weights_path = "exported_c/weights.h"
    images_path = "exported_c/test_images.h"
    
    print("[*] Parsing C-Headers...")
    # Extract weights. W_linear is exported as 16x16 with padding, so we slice the first 8 columns.
    W_conv = extract_c_array(weights_path, "W_conv").reshape((16, 16))
    W_linear = extract_c_array(weights_path, "W_linear").reshape((16, 16))[:, :8]
    
    anchors = {
        0: extract_c_array(weights_path, "Anchor_Person_0"),
        1: extract_c_array(weights_path, "Anchor_Person_1"),
        2: extract_c_array(weights_path, "Anchor_Person_2")
    }
    
    image_names = get_test_image_names(images_path)
    print(f"[*] Found {len(image_names)} test images. Commencing Integer Inference...\n")
    
    THRESHOLD = 1.7
    
    for img_name in image_names:
        # Extract the true label from the C-array name (e.g., "Person_2_Test_1" -> 2)
        true_person_id = int(img_name.split('_')[1])
        
        # Load the 32x32 image from the header
        img_array = extract_c_array(images_path, img_name).reshape((32, 32))
        
        # Run exactly what the Nios II will run
        live_signature = hardware_inference(img_array, W_conv, W_linear)
        
        print(f"--- Testing {img_name} ---")
        best_match_id = -1
        lowest_dist = float('inf')
        
        # Compare against all 3 whitelisted anchors
        for anchor_id, anchor_vec in anchors.items():
            dist = calc_pytorch_distance(live_signature, anchor_vec)
            print(f"  Distance to Anchor {anchor_id}: {dist:.2f}")
            
            if dist < lowest_dist:
                lowest_dist = dist
                best_match_id = anchor_id
                
        # Access Control Logic
        if lowest_dist < THRESHOLD:
            if best_match_id == true_person_id:
                print(f"  [SUCCESS] Access Granted to Person {best_match_id}. (Correct Identity)")
            else:
                print(f"  [FAILURE] Access Granted to Person {best_match_id}. (FALSE ACCEPTANCE!)")
        else:
            print(f"  [DENIED] Face not recognized. Distance {lowest_dist:.2f} > {THRESHOLD}")
        print("")