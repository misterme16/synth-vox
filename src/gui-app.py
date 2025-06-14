import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import threading
import traceback

# Dependencies (you'll need to define these functions)
from processing_modules import run_vocoder, run_talkbox, convert_to_midi  # See below

class TalkboxFrame(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="Talkbox Parameters", padding=10)
        
        # Create parameter variables
        self.drive = tk.DoubleVar(value=2.5)
        self.wah_min_freq = tk.DoubleVar(value=400)
        self.wah_max_freq = tk.DoubleVar(value=2000)
        self.wah_speed = tk.DoubleVar(value=3.0)
        self.wah_depth = tk.DoubleVar(value=0.8)
        self.wah_resonance = tk.DoubleVar(value=8.0)
        self.pre_emphasis_freq = tk.DoubleVar(value=1200)
        self.compression_ratio = tk.DoubleVar(value=6.0)
        self.compression_attack = tk.DoubleVar(value=0.005)
        self.compression_release = tk.DoubleVar(value=0.15)
        
        # Store slider references using variable names as keys
        self.sliders = {}
        
        # Create and pack the parameter controls
        self.create_slider("Drive (Distortion)", self.drive, 1.0, 8.0, 0.1, "drive")
        
        # Wah controls
        wah_frame = ttk.LabelFrame(self, text="Wah Effect", padding=5)
        wah_frame.pack(fill='x', pady=5)
        
        self.create_slider("Min Freq (Hz)", self.wah_min_freq, 100, 1000, 10, "wah_min_freq", parent=wah_frame)
        self.create_slider("Max Freq (Hz)", self.wah_max_freq, 1000, 5000, 100, "wah_max_freq", parent=wah_frame)
        self.create_slider("Speed (Hz)", self.wah_speed, 0.5, 10.0, 0.1, "wah_speed", parent=wah_frame)
        self.create_slider("Depth", self.wah_depth, 0.0, 1.0, 0.01, "wah_depth", parent=wah_frame)
        self.create_slider("Resonance", self.wah_resonance, 1.0, 20.0, 0.1, "wah_resonance", parent=wah_frame)
        
        # Tone controls
        tone_frame = ttk.LabelFrame(self, text="Tone Shaping", padding=5)
        tone_frame.pack(fill='x', pady=5)
        
        self.create_slider("Pre-emphasis (Hz)", self.pre_emphasis_freq, 500, 2000, 100, "pre_emphasis_freq", parent=tone_frame)
        
        # Compression controls
        comp_frame = ttk.Frame(self)
        comp_frame.pack(fill='x', pady=5)
        
        ttk.Label(comp_frame, text="Compression", font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', pady=(5,0))
        
        comp_controls = ttk.Frame(comp_frame, padding=5)
        comp_controls.pack(fill='x', expand=True)
        
        self.create_slider("Ratio", self.compression_ratio, 1.0, 10.0, 0.1, "compression_ratio", parent=comp_controls)
        self.create_slider("Attack (ms)", self.compression_attack, 1, 50, 1, "compression_attack",
                         parent=comp_controls, value_transform=lambda x: x/1000)
        self.create_slider("Release (ms)", self.compression_release, 50, 500, 10, "compression_release",
                         parent=comp_controls, value_transform=lambda x: x/1000)
        
        # Preset buttons
        preset_frame = ttk.LabelFrame(self, text="Presets", padding=5)
        preset_frame.pack(fill='x', pady=5)
        
        # Create a grid frame for preset buttons
        button_frame = ttk.Frame(preset_frame)
        button_frame.pack(fill='x', expand=True)
        
        # Add preset buttons in a grid layout
        ttk.Button(button_frame, text="Heavy Metal", 
                  command=lambda: self.load_preset("metal")).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(button_frame, text="Blues/Rock", 
                  command=lambda: self.load_preset("blues")).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(button_frame, text="Clean Jazz", 
                  command=lambda: self.load_preset("jazz")).grid(row=0, column=2, padx=2, pady=2)
        ttk.Button(button_frame, text="Max Wah", 
                  command=lambda: self.load_preset("wah")).grid(row=0, column=3, padx=2, pady=2)
        
        # Configure grid columns to expand evenly
        for i in range(4):
            button_frame.grid_columnconfigure(i, weight=1)
    
    def create_slider(self, label, variable, min_val, max_val, step, var_name, parent=None, value_transform=None):
        if parent is None:
            parent = self
        
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=2)
        
        # Label frame to hold both label and value
        label_frame = ttk.Frame(frame)
        label_frame.pack(side='left', padx=(0, 10))
        
        ttk.Label(label_frame, text=label).pack(side='left')
        value_label = ttk.Label(label_frame, width=8)
        value_label.pack(side='left', padx=(5, 0))
        
        if value_transform is None:
            value_transform = lambda x: x
            inverse_transform = lambda x: x
        else:
            inverse_transform = lambda x: x * 1000
        
        def update_value(val):
            try:
                float_val = float(val)
                transformed_val = value_transform(float_val)
                variable.set(transformed_val)
                
                # Format display value
                if callable(value_transform) and value_transform.__name__ == '<lambda>' and value_transform(1.0) == 0.001:
                    display_val = float_val  # Show in ms
                else:
                    display_val = transformed_val
                    
                # Format with appropriate precision
                if display_val < 10:
                    value_label.config(text=f"{display_val:.2f}")
                else:
                    value_label.config(text=f"{display_val:.1f}")
            except ValueError:
                pass
        
        slider = ttk.Scale(frame, from_=min_val, to=max_val, 
                          value=inverse_transform(variable.get()),
                          orient='horizontal', command=update_value)
        slider.pack(side='right', fill='x', expand=True)
        
        # Store slider reference with its transform function using variable name as key
        self.sliders[var_name] = (slider, inverse_transform)
        
        # Initialize value label
        update_value(inverse_transform(variable.get()))
    
    def load_preset(self, preset):
        preset_values = {
            "metal": {
                'drive': 6.0,
                'pre_emphasis_freq': 1500,
                'compression_ratio': 8.0,
                'wah_resonance': 10.0,
                'wah_depth': 0.7,
                'wah_min_freq': 400,
                'wah_max_freq': 2000,
                'wah_speed': 3.0,
                'compression_attack': 0.005,
                'compression_release': 0.15
            },
            "blues": {
                'drive': 3.0,
                'pre_emphasis_freq': 1000,
                'wah_min_freq': 300,
                'wah_max_freq': 2500,
                'compression_ratio': 5.0,
                'wah_resonance': 6.0,
                'wah_depth': 0.6,
                'wah_speed': 2.0,
                'compression_attack': 0.01,
                'compression_release': 0.2
            },
            "jazz": {
                'drive': 1.5,
                'pre_emphasis_freq': 800,
                'wah_depth': 0.5,
                'compression_ratio': 4.0,
                'wah_resonance': 6.0,
                'wah_min_freq': 200,
                'wah_max_freq': 1500,
                'wah_speed': 1.5,
                'compression_attack': 0.015,
                'compression_release': 0.25
            },
            "wah": {
                'drive': 4.0,
                'wah_min_freq': 200,
                'wah_max_freq': 3000,
                'wah_speed': 4.0,
                'wah_depth': 0.9,
                'wah_resonance': 12.0,
                'pre_emphasis_freq': 1200,
                'compression_ratio': 6.0,
                'compression_attack': 0.005,
                'compression_release': 0.15
            }
        }
        
        if preset in preset_values:
            for var_name, value in preset_values[preset].items():
                var = getattr(self, var_name)
                var.set(value)
                # Update slider position
                if var_name in self.sliders:
                    slider, transform = self.sliders[var_name]
                    slider.set(transform(value))
    
    def get_parameters(self):
        return {
            'drive': self.drive.get(),
            'wah_min_freq': self.wah_min_freq.get(),
            'wah_max_freq': self.wah_max_freq.get(),
            'wah_speed': self.wah_speed.get(),
            'wah_depth': self.wah_depth.get(),
            'wah_resonance': self.wah_resonance.get(),
            'pre_emphasis_freq': self.pre_emphasis_freq.get(),
            'compression_ratio': self.compression_ratio.get(),
            'compression_attack': self.compression_attack.get(),
            'compression_release': self.compression_release.get()
        }

class VocoderFrame(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="Vocoder Parameters", padding=10)
        
        # Create parameter variables
        self.formant_shift = tk.DoubleVar(value=1.0)  # 0.5 to 2.0
        self.spectral_envelope = tk.DoubleVar(value=1.0)  # 0.5 to 2.0
        self.frequency_range = tk.DoubleVar(value=1.0)  # 0.5 to 2.0
        self.aperiodicity = tk.DoubleVar(value=0.5)  # 0.0 to 1.0
        self.clarity = tk.DoubleVar(value=0.7)  # 0.0 to 1.0
        
        # Store slider references using variable names as keys
        self.sliders = {}
        
        # Create and pack the parameter controls
        self.create_slider("Formant Shift", self.formant_shift, 0.5, 2.0, 0.1, "formant_shift")
        self.create_slider("Spectral Envelope", self.spectral_envelope, 0.5, 2.0, 0.1, "spectral_envelope")
        self.create_slider("Frequency Range", self.frequency_range, 0.5, 2.0, 0.1, "frequency_range")
        self.create_slider("Aperiodicity", self.aperiodicity, 0.0, 1.0, 0.1, "aperiodicity")
        self.create_slider("Clarity", self.clarity, 0.0, 1.0, 0.1, "clarity")
        
        # Add preset buttons
        preset_frame = ttk.LabelFrame(self, text="Presets", padding=5)
        preset_frame.pack(fill='x', pady=5)
        
        # Create a grid frame for preset buttons
        button_frame = ttk.Frame(preset_frame)
        button_frame.pack(fill='x', expand=True)
        
        # Add preset buttons in a grid layout
        ttk.Button(button_frame, text="Original", 
                  command=lambda: self.load_preset("original")).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(button_frame, text="Robot", 
                  command=lambda: self.load_preset("robot")).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(button_frame, text="Alien", 
                  command=lambda: self.load_preset("alien")).grid(row=0, column=2, padx=2, pady=2)
        ttk.Button(button_frame, text="Deep Voice", 
                  command=lambda: self.load_preset("deep")).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(button_frame, text="High Voice", 
                  command=lambda: self.load_preset("high")).grid(row=1, column=1, padx=2, pady=2)
        
        # Configure grid columns to expand evenly
        for i in range(3):
            button_frame.grid_columnconfigure(i, weight=1)
    
    def create_slider(self, label, variable, min_val, max_val, step, var_name):
        frame = ttk.Frame(self)
        frame.pack(fill='x', pady=2)
        
        # Label frame to hold both label and value
        label_frame = ttk.Frame(frame)
        label_frame.pack(side='left', padx=(0, 10))
        
        ttk.Label(label_frame, text=label).pack(side='left')
        value_label = ttk.Label(label_frame, width=8)
        value_label.pack(side='left', padx=(5, 0))
        
        def update_value(val):
            try:
                float_val = float(val)
                variable.set(float_val)
                value_label.config(text=f"{float_val:.2f}")
            except ValueError:
                pass
        
        slider = ttk.Scale(frame, from_=min_val, to=max_val, 
                          value=variable.get(),
                          orient='horizontal', command=update_value)
        slider.pack(side='right', fill='x', expand=True)
        
        # Store slider reference using variable name as key
        self.sliders[var_name] = slider
        
        # Initialize value label
        update_value(variable.get())
    
    def load_preset(self, preset):
        preset_values = {
            "original": {
                'formant_shift': 1.0,
                'spectral_envelope': 1.0,
                'frequency_range': 1.0,
                'aperiodicity': 0.5,
                'clarity': 0.7
            },
            "robot": {
                'formant_shift': 1.0,
                'spectral_envelope': 1.2,
                'frequency_range': 1.5,
                'aperiodicity': 0.8,
                'clarity': 0.9
            },
            "alien": {
                'formant_shift': 1.8,
                'spectral_envelope': 1.5,
                'frequency_range': 1.8,
                'aperiodicity': 0.6,
                'clarity': 0.5
            },
            "deep": {
                'formant_shift': 0.85,
                'spectral_envelope': 1.1,
                'frequency_range': 0.9,
                'aperiodicity': 0.4,
                'clarity': 0.75
            },
            "high": {
                'formant_shift': 1.5,
                'spectral_envelope': 0.8,
                'frequency_range': 1.2,
                'aperiodicity': 0.4,
                'clarity': 0.9
            }
        }
        
        if preset in preset_values:
            for var_name, value in preset_values[preset].items():
                var = getattr(self, var_name)
                var.set(value)
                # Update slider position
                if var_name in self.sliders:
                    self.sliders[var_name].set(value)
    
    def get_parameters(self):
        return {
            'formant_shift': self.formant_shift.get(),
            'spectral_envelope': self.spectral_envelope.get(),
            'frequency_range': self.frequency_range.get(),
            'aperiodicity': self.aperiodicity.get(),
            'clarity': self.clarity.get()
        }

class ProcessingControls(ttk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="Processing Controls", padding=10)
        
        # Create variables
        self.processing_type = tk.StringVar(value="talkbox")
        
        # Create radio buttons for processing type
        ttk.Radiobutton(self, text="Talkbox Effect", 
                       variable=self.processing_type, 
                       value="talkbox").pack(anchor='w')
                       
        ttk.Radiobutton(self, text="Vocoder", 
                       variable=self.processing_type, 
                       value="vocoder").pack(anchor='w')
                       
        ttk.Radiobutton(self, text="MIDI Export", 
                       variable=self.processing_type, 
                       value="synth").pack(anchor='w')

# === GUI Application ===
class SynthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Synth-Vox")
        
        # Configure main window
        self.root.minsize(500, 600)  # Increased minimum height
        self.root.geometry("600x900")  # Increased default height
        
        # Create main canvas with scrollbar
        main_canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        
        # Create main frame inside canvas
        main_frame = ttk.Frame(main_canvas, padding="10")
        
        # Configure scrolling
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        main_canvas.pack(side="left", fill="both", expand=True)
        
        # Create window in canvas for the main frame
        canvas_frame = main_canvas.create_window((0, 0), window=main_frame, anchor="nw", width=main_canvas.winfo_width())
        
        # Update canvas scroll region when frame size changes
        def configure_scroll_region(event):
            main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        
        def configure_canvas_width(event):
            main_canvas.itemconfig(canvas_frame, width=event.width)
        
        main_frame.bind("<Configure>", configure_scroll_region)
        main_canvas.bind("<Configure>", configure_canvas_width)
        
        # Enable mousewheel scrolling
        def on_mousewheel(event):
            main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        main_canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding=10)
        file_frame.pack(fill='x', pady=(0, 10))
        
        # Input file selection
        input_frame = ttk.Frame(file_frame)
        input_frame.pack(fill='x', pady=2)
        ttk.Label(input_frame, text="Input Audio:").pack(side='left')
        self.input_label = ttk.Label(input_frame, text="No file selected")
        self.input_label.pack(side='left', padx=5)
        ttk.Button(input_frame, text="Browse", command=self.select_input).pack(side='right')
        
        # Output file selection
        output_frame = ttk.Frame(file_frame)
        output_frame.pack(fill='x', pady=2)
        ttk.Label(output_frame, text="Output File:").pack(side='left')
        self.output_label = ttk.Label(output_frame, text="No file selected")
        self.output_label.pack(side='left', padx=5)
        ttk.Button(output_frame, text="Browse", command=self.select_output).pack(side='right')
        
        # Processing controls frame
        self.processing_controls = ProcessingControls(main_frame)
        self.processing_controls.pack(fill='x', pady=(0, 10))
        
        # Add trace to processing_type to show/hide parameter frames
        self.processing_controls.processing_type.trace('w', self._on_processing_type_change)
        
        # Talkbox parameters frame
        self.talkbox_frame = TalkboxFrame(main_frame)
        
        # Vocoder parameters frame
        self.vocoder_frame = VocoderFrame(main_frame)
        
        # Run button and status
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=10)
        
        self.run_button = ttk.Button(control_frame, text="Process Audio", command=self.process_audio)
        self.run_button.pack(side='left')
        
        self.status_label = ttk.Label(control_frame, text="Ready")
        self.status_label.pack(side='left', padx=10)
        
        # Store file paths
        self.input_file = None
        self.output_file = None
        
        # Initial UI update
        self._on_processing_type_change()
    
    def _on_processing_type_change(self, *args):
        # Show/hide parameter frames based on processing type
        processing_type = self.processing_controls.processing_type.get()
        
        # Hide all parameter frames first
        self.talkbox_frame.pack_forget()
        self.vocoder_frame.pack_forget()
        
        # Show the appropriate frame
        if processing_type == "talkbox":
            self.talkbox_frame.pack(fill='x', pady=(0, 10))
        elif processing_type == "vocoder":
            self.vocoder_frame.pack(fill='x', pady=(0, 10))

    def select_input(self):
        self.input_file = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if self.input_file:
            self.input_label.config(text=self.input_file)
            
            # Only suggest output filename if none has been selected yet
            if not self.output_file:
                dir_name = os.path.dirname(self.input_file)
                base_name = os.path.splitext(os.path.basename(self.input_file))[0]
                processing_type = self.processing_controls.processing_type.get()
                
                if processing_type == "synth":
                    ext = ".mid"
                    suffix = "_midi"
                else:
                    ext = ".wav"
                    suffix = f"_{processing_type}"
                    
                # Create suggested filename in same directory but with different name
                suggested_output = os.path.join(dir_name, f"{base_name}{suffix}{ext}")
                
                # If file exists, add number to make unique
                counter = 1
                while os.path.exists(suggested_output):
                    suggested_output = os.path.join(dir_name, f"{base_name}{suffix}_{counter}{ext}")
                    counter += 1
                    
                self.output_label.config(text="Click Browse to select output location")

    def select_output(self):
        processing_type = self.processing_controls.processing_type.get()
        if processing_type == "synth":
            self.output_file = filedialog.asksaveasfilename(
                defaultextension=".mid",
                filetypes=[("MIDI files", "*.mid")]
            )
        else:
            self.output_file = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav")]
            )
        if self.output_file:
            self.output_label.config(text=self.output_file)

    def process_audio(self):
        """Process the audio file with selected effect"""
        if not self.input_file or not os.path.exists(self.input_file):
            self.status_label.config(text="Error: Please select an input file first")
            return

        if not self.output_file:
            self.status_label.config(text="Error: Please select an output file")
            return

        try:
            self.status_label.config(text="Processing...")
            self.root.update()
            
            processing_type = self.processing_controls.processing_type.get()
            
            if processing_type == "talkbox":
                run_talkbox(
                    self.input_file, 
                    self.output_file,
                    drive=self.talkbox_frame.drive.get(),
                    wah_min_freq=self.talkbox_frame.wah_min_freq.get(),
                    wah_max_freq=self.talkbox_frame.wah_max_freq.get(),
                    wah_speed=self.talkbox_frame.wah_speed.get(),
                    wah_depth=self.talkbox_frame.wah_depth.get(),
                    wah_resonance=self.talkbox_frame.wah_resonance.get(),
                    pre_emphasis_freq=self.talkbox_frame.pre_emphasis_freq.get(),
                    compression_ratio=self.talkbox_frame.compression_ratio.get(),
                    compression_attack=self.talkbox_frame.compression_attack.get(),
                    compression_release=self.talkbox_frame.compression_release.get()
                )
            elif processing_type == "vocoder":
                run_vocoder(
                    self.input_file, 
                    self.output_file,
                    formant_shift=self.vocoder_frame.formant_shift.get(),
                    spectral_envelope=self.vocoder_frame.spectral_envelope.get(),
                    frequency_range=self.vocoder_frame.frequency_range.get(),
                    aperiodicity=self.vocoder_frame.aperiodicity.get(),
                    clarity=self.vocoder_frame.clarity.get()
                )
            else:  # synth mode
                convert_to_midi(self.input_file, self.output_file)
                
            self.status_label.config(text=f"Done! Saved to: {self.output_file}")
        
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            
        self.root.update()

# === Run App ===
if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')  # Use a modern theme
    app = SynthApp(root)
    root.mainloop()
