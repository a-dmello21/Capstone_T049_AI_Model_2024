import sys
import tensorflow as tf




# Check if the script is being called with arguments
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error - More then one Parameter provided")
        sys.exit(1)

    paramater = sys.argv[1]

    # Load the .h5 model
    model = tf.keras.models.load_model('./trained_model.h5')

    # Print the model summary to check its architecture
    model.summary()
    