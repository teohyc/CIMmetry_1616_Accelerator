import math

def generate_swish_hex(filename="swish_values.hex", scale=16.0):
    with open(filename, 'w') as f:
        print(f"Generating LUT for Swish function... (Scale: {scale})")
        
        #loop through -128 to 127
        for i in range(-128, 128):
            
            #convert the hardware integer to a floating-point number
            x_float = i / scale
            
            #swish: x * sigmoid(x)
            sigmoid = 1.0 / (1.0 + math.exp(-x_float))
            swish_float = x_float * sigmoid
            
            #convert the float back to hardware integer space
            swish_int = round(swish_float * scale)
            
            #saturation clamp
            if swish_int > 127:
                swish_int = 127
            elif swish_int < -128:
                swish_int = -128
                
            #mask with 0xFF to handle 2s complement negative numbers
            hex_val = swish_int & 0xFF
            
            #write to file, padding to exactly 2 uppercase hex digits per line
            f.write(f"{hex_val:02X}\n")
            
    print(f"{filename} is ready to be loaded into Quartus.")

if __name__ == "__main__":
    generate_swish_hex()
        