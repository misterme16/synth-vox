import pedalboard
from pedalboard import Pedalboard, Chorus, Reverb

print("Pedalboard version:", pedalboard.__version__)

# Create a simple pedalboard
board = Pedalboard([
    Chorus(),
    Reverb(room_size=0.5)
])

print("Successfully created pedalboard:", board) 