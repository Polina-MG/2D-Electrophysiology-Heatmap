# R&D Internship
# January-March 2025
# -- The University of Tokyo, Institute of Industrial Sciences / LIMMS --
# -- Tixier-Mita laboratory --


# =============== LIBRARY IMPORTATION ==============


# libraries included in base python
import numpy as np #https://numpy.org/doc/stable/reference/index.html#reference
import sys #https://docs.python.org/3/library/sys.html
import os #https://docs.python.org/3/library/os.html, to check if file exists or not
import multiprocessing as mp #https://docs.python.org/3/library/multiprocessing.html
from itertools import chain #https://docs.python.org/3/library/itertools.html, used for mean of list (mean_peak_per_electrode) calculation method from https://www.geeksforgeeks.org/python-average-of-two-lists/


# libraries to import
import matplotlib.pyplot as plt #https://matplotlib.org/stable/api/pyplot_summary.html
import pyqtgraph as pg #https://pyqtgraph.readthedocs.io/en/latest/index.html
from PyQt5.QtCore import Qt, pyqtSignal #https://doc.qt.io/qtforpython-5/PySide2/QtCore/Qt.html, documentation from Qt is quasi-identical to PyQt5 according to https://stackoverflow.com/questions/60422323/where-is-the-pyqt5-documentation-for-classes-methods-and-modules
from PyQt5.QtGui import QFont, QImage, QPainter #https://doc.qt.io/qtforpython-5/PySide2/QtGui/index.html
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QGroupBox, QGridLayout, QAction, QWidget, QPushButton, QLabel, QLineEdit, QFileDialog, QSlider, QComboBox, QMessageBox, QFrame
   #https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/index.html
   # for menu bar, https://realpython.com/python-menus-toolbars/#populating-menus-with-actions
import pandas as pd #https://pandas.pydata.org/docs/user_guide/index.html
from scipy.signal import find_peaks, butter, sosfiltfilt
   # find_peaks https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
   # butter https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
   # sosfiltfilt https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html


# =================== FILTER FUNCTIONS ===================
def apply_highpass(signal, sampling_rate, lowcut):
    """
    Apply a highpass filter to the signal. It removes low-frequency noise and straightens the signal.
    """
    nyq = 0.5 * sampling_rate # nyq = nyquist frequency
    if lowcut:
        hp_sos = butter(2, lowcut / nyq, btype='highpass', output='sos')
        signal = sosfiltfilt(hp_sos, signal)
    return signal


# =================== SEGMENTATION FUNCTION ===================
def process_column(args):
    """
    Process segmentation of one column (=one drain line) of the DataFrame
    and save the segmented data to a new file with one column by  electrode.
    """
    col_name, signal, output_dir, sampling_rate, pulse_time, delay_time, lowcut, beginning_index= args
    signal = pd.to_numeric(signal, errors="coerce").astype("float32").values

    # parameters
    pulse_time = int(pulse_time*sampling_rate/1000) # length of indexes, not in ms
    delay_time = int(delay_time*sampling_rate/1000) # length of indexes, not in ms
    seg_offset = int(750*sampling_rate/1000) # 750ms offset to avoid overlapping, seg_offset is not in ms
    seg_interval = pulse_time - int(500*sampling_rate/1000) # length of points to add for a segment, -500ms to avoid overlapping if user mischoose the beginning_index, seg_interval is not in ms

    # electrode segmentation by time
    segments = [signal[beginning_index + seg_offset + i*(pulse_time+delay_time) : beginning_index + seg_interval + i*(pulse_time+delay_time)] for i in range(0,28)] # contains list for each separated electrode (by time)
    max_len = max(len(seg) for seg in segments)
    segments = [np.pad(seg, (0, max_len - len(seg)), mode='constant', constant_values=np.nan) for seg in segments]
        #https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    # filtering after segmentation
    filtered_segments = [apply_highpass (seg, sampling_rate, lowcut) for seg in segments] # straighten the signal

    # save to txt file
    df_out = pd.DataFrame({f"Electrode_{i+1}": seg for i, seg in enumerate(filtered_segments)}) # df_out is a Pandas dataframe, where "Electrode_{i+1}" is a column of corresponding i segment
    df_out.to_csv(os.path.join(output_dir, f"DrainLine_{col_name}.txt"), sep="\t", index=False) # df_out is named DrainLine_{col_name} with col_name the name of the corresponding drain line
    print(f"{col_name} processed.")
    return col_name, pd.DataFrame({f"Electrode_{i+1}": seg for i, seg in enumerate(filtered_segments)})


# =================== THREADING FUNCTION ===================
def process_all_columns(df, output_dir, sampling_rate, pulse_time, delay_time, lowcut, beginning_index):
    """
    Apply segmentation function for every column of the DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)
    with mp.Pool(processes=mp.cpu_count()) as pool: # creates a pool of worker processes with number of cpu cores (mp.cpu_count())
        results = pool.map(process_column, [(col, df[col], output_dir, sampling_rate, pulse_time, delay_time, lowcut, beginning_index) for col in df.columns])
    return {col: df for col, df in results if df is not None}
        # runs one column on one worker process,
        # so each worker process works on different columns at the same time


# =================== HEATMAP FUNCTION ===================
def generate_heatmap_data(self, processed_data, threshold_factor):
    """
    Generate heatmap for each electrode based on the processed data.
    One heatmap is generated for each of the four parameters: Amplitude Peak to Peak, Sign, Frequency, and Propagation.
    """
    # dictionnary, to keep each corresponding electrode value
    heatmaps = {
        "Amplitude Peak to Peak": [], # mean amplitude peak to peak, in uV
        "Sign": [], # amplitude sign, in uV
        "Frequency": [], # in Hz
        "Propagation": [] # first peak time for each electrode, ms, shortest will be red
    }

    for df in processed_data.values():
        mean_peak_per_row = [] # list to keep mean amplitude value for each electrode in one row
        amplitude_sign_per_row = [] # list to keep amplitude sign value for each electrode in one row
        frequency_per_row = [] # list to keep frequency value for each electrode in one row
        firstpeaktime_per_row = [] # list to keep first peak time value for each electrode in one row
        for col in df.columns:
            column_data = df[col].dropna().values # drop NaN values
            if len(column_data) == 0:
                mean_peak_per_row.append(np.nan) # if list empty, fill with NaN and continue
                continue
            # dynamic threshold, threshold factor can be modified from 1 to 100 by user
            threshold_activity = 0.1 * threshold_factor * np.mean(np.abs(column_data)) # threshold_factor can be modified by user, with a slider on HeatmapWindow
            # detection of positive peaks (posi_peaks), negative peaks (nega_peaks)
            posi_peaks, posi_props = find_peaks(column_data, height=threshold_activity, distance=int(100*self.sampling_rate/1000))
            nega_peaks, nega_props = find_peaks(-column_data, height=threshold_activity, distance=int(100*self.sampling_rate/1000))
            mean_peak_per_electrode = sum(chain(posi_props['peak_heights'], nega_props['peak_heights'])) / max(1, len(posi_props['peak_heights']) + len(nega_props['peak_heights']))
                # mean calculation method: https://www.geeksforgeeks.org/python-average-of-two-lists/
            amplitude_sign_per_electrode = sum(posi_props['peak_heights']) - sum(nega_props['peak_heights'])
            period_list_per_electrode = np.diff(posi_peaks) if len(np.diff(posi_peaks))>len(np.diff(nega_peaks)) else np.diff(nega_peaks) # list of most detected peaks pattern between positive or negative
                    # np.diff: https://numpy.org/doc/2.1/reference/generated/numpy.diff.html
            if not(period_list_per_electrode.any()): # True if empty
                frequency_per_electrode = 0
            else:
                frequency_per_electrode = ( 1 / len(period_list_per_electrode) ) * sum( 1 / (T_i/(self.sampling_rate)) for T_i in period_list_per_electrode )  # for sampling_rate=10000Hz, one indice = 0.1ms = 0.0001s
            if len(posi_peaks) > 0 and len(nega_peaks) > 0:
                firstpeaktime_per_electrode = min(posi_peaks[0], nega_peaks[0]) / 10 # measures the signal propagation
            else:
                firstpeaktime_per_electrode = np.nan  # or any other default value
            # add values to row list
            mean_peak_per_row.append(mean_peak_per_electrode)
            amplitude_sign_per_row.append(amplitude_sign_per_electrode)
            frequency_per_row.append(frequency_per_electrode)
            firstpeaktime_per_row.append(firstpeaktime_per_electrode)
        # add value to heatmaps dictionnary
        heatmaps["Amplitude Peak to Peak"].append(mean_peak_per_row)
        heatmaps["Sign"].append(amplitude_sign_per_row)
        heatmaps["Frequency"].append(frequency_per_row)
        heatmaps["Propagation"].append(firstpeaktime_per_row)
    # dataframe conversion
    heatmap_dfs = {
        key: pd.DataFrame(data, columns=[f"Electrode_{i+1}" for i in range(df.shape[1])]).fillna(0)
        for key, data in heatmaps.items()
    }
    return heatmap_dfs


# =================== GUI WINDOW ===================


# 2.1.2. DATA PROCESSING PARAMETERS WINDOW
class DataProcessingWindow(QWidget):
    update_filename_signal = pyqtSignal(str) # signal to send filename to main window HeatmapWindow
    update_all_parameters = pyqtSignal(int, list, list, int) # signal to send
        # int:sampling rate (Hz);
        # list: scan list method;
        # list: dictionnary of numbers of drain lines;
        # int: beginning index of slicing.


    def __init__(self):
        """
        Data processing window initialization.
        """
        # initialization
        super().__init__()
        self.setWindowTitle("Data Processing Parameters")
        self.file_name = ""
        self.beginning_index = 0
        self.tracking = False  # track mouse movement, initially locked
        # main data processing layout
        datapross_layout = QVBoxLayout() # vertical layout

        # browse layout
        self.browse_box = QGroupBox("Select a txt file") # group box layout
        browse_layout = QHBoxLayout() # horizontal layout
        # line to paste path of file
        self.file_input = QLineEdit() # line input
        self.file_input.setPlaceholderText("Enter the file path or click 'Browse'...")
        browse_layout.addWidget(self.file_input)
        # browse button
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)
        browse_layout.addWidget(self.browse_button)
        self.browse_box.setLayout(browse_layout)
        datapross_layout.addWidget(self.browse_box)

        # sampling rate box input
        self.samplrate_box = QGroupBox("MCS sampling rate (Hz)")
        samplrate_layout = QVBoxLayout()
        self.samplrate_input = QLineEdit()
        self.samplrate_input.setText('10000')
        samplrate_layout.addWidget(self.samplrate_input)
        self.samplrate_box.setLayout(samplrate_layout)
        datapross_layout.addWidget(self.samplrate_box)

        # SCAN method box inputs
        self.scanmethod_box = QGroupBox("SCAN method")
        scanmethod_layout = QGridLayout() # grid layout
        self.segtime_label = QLabel('Pulse (ms)')
        self.segtime_input = QLineEdit()
        self.segtime_input.setText('5000')
        self.pausetime_label = QLabel('Delay (ms)')
        self.pausetime_input = QLineEdit()
        self.pausetime_input.setText('1000')
        scanmethod_layout.addWidget(self.segtime_label, 1,1, alignment=Qt.AlignBottom)
        scanmethod_layout.addWidget(self.segtime_input, 2,1)
        scanmethod_layout.addWidget(self.pausetime_label, 1,2, alignment=Qt.AlignBottom)
        scanmethod_layout.addWidget(self.pausetime_input, 2,2)
        self.scanmethod_box.setLayout(scanmethod_layout)
        datapross_layout.addWidget(self.scanmethod_box)

        # drain lines box buttons
        self.drain_buttons = {} # dictionnary to store number line
        self.drainlines_box = QGroupBox("Select Drain Lines to remove") # group box layout
        drainlines_layout = QGridLayout() # grid layout
        for i in range (1,33): # 32 buttons
            btn = QPushButton(str(i)) # button with i number
            btn.setCheckable(True)
            btn.clicked.connect(self.remove_drain)
            row, col = divmod(i - 1, 8)
            drainlines_layout.addWidget(btn, row, col)
            self.drain_buttons[i] = btn
        self.drainlines_box.setLayout(drainlines_layout)
        datapross_layout.addWidget(self.drainlines_box)

        # horizontal separator
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.HLine)
        self.separator.setFrameShadow(QFrame.Sunken)
        datapross_layout.addWidget(self.separator)

        # SCAN check beginning button
        self.scanbegin_button = QPushButton("Check SCAN beginnning")
        self.scanbegin_button.clicked.connect(self.datapross_plot_beginning_index)
        datapross_layout.addWidget(self.scanbegin_button)
        # label for instructions
        self.instruction_label = QLabel("Select the first spike by sliding the red vertical bar below:")
        datapross_layout.addWidget(self.instruction_label)
        # plot widget
        self.datapross_plot = pg.PlotWidget()
        self.datapross_plot.setMinimumSize(400, 200)
        self.datapross_plot.scene().sigMouseClicked.connect(self.on_click)
        self.datapross_plot.scene().sigMouseMoved.connect(self.on_mouse_move)
        # vertical line
        self.v_line = pg.InfiniteLine(angle=90, movable=True, pen='r')
        self.datapross_plot.addItem(self.v_line)
        datapross_layout.addWidget(self.datapross_plot)

        # horizontal separator
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.HLine)
        self.separator.setFrameShadow(QFrame.Sunken)
        datapross_layout.addWidget(self.separator)

        # process data button
        self.processdata_label = QLabel("Choose the repertory where you want to save the processed data.")
        self.load_button = QPushButton("Process data", self)
        self.load_button.clicked.connect(self.close_and_update_main)
        datapross_layout.addWidget(self.processdata_label, alignment=Qt.AlignBottom)
        datapross_layout.addWidget(self.load_button)
        self.setLayout(datapross_layout)


    # =============== ACTION FUNCTIONS ===============  
    def browse_file(self): # browse file function
        file_name, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Text Files (*.txt);;All Files (*)") # opens window to choose txt file repertory
        if file_name:
            self.file_input.setText(file_name)


    def datapross_plot_beginning_index(self):
        """
        Function to check the first spike to start segmentation with.
        """
        file_name = self.file_input.text() # retrieve file_name from file_input
        sampling_rate = int(self.samplrate_input.text())
        pulse_time = int(self.segtime_input.text())
        delay_time = int(self.pausetime_input.text())
        selected_drains = [key for key, btn in self.drain_buttons.items() if btn.isChecked()]
        removed_drains = [i for i in range(1,33) if i not in selected_drains]

        # read drain line data
        df = pd.read_csv(file_name, delimiter="\t", encoding="iso-8859-1", skiprows=2, low_memory=False, dtype=str)
        time = pd.to_numeric(df[df.columns[0]][0:int(30000*sampling_rate/1000)], errors='coerce') # list of 30000ms = 30s
        beginning_signal = pd.to_numeric(df[df.columns[removed_drains[0]]][0:int(30000*sampling_rate/1000)], errors='coerce')
            # beginning_signal = first drain line not included in selected_drains
            # list of 30000ms = 30s

        # threshold
        mean_abs_beginning_signal = np.nanmean(np.abs(beginning_signal)) # mean of all non NaN values
        threshold_beginning_gate = 10 * mean_abs_beginning_signal
        # beginning peak calculation
        beginning_peaks, _ = find_peaks(-beginning_signal, height=threshold_beginning_gate, distance=int((pulse_time+delay_time)*sampling_rate/1000))

        # plotting with data
        self.datapross_plot.clear() # clear previous plot
        drain_line_title = df.columns[removed_drains[0]] # column name for drain line, "Ei 0x" for raw data, "Fi 0x" for filtered data
        self.datapross_plot.setTitle(f"Drain line: {drain_line_title}")
        self.datapross_plot.setLabel("bottom", "Time (ms)") # y-axis
        self.datapross_plot.setLabel("left", "Amplitude (mV)") # x-axis
        self.datapross_plot.addItem(self.v_line) # add red vertical line
        self.datapross_plot.plot(time, beginning_signal) # plot beginning_signal vs time
        # reset vertical line position
        self.beginning_index = beginning_peaks[0] # index of the first spike
        if 0 <= self.beginning_index < len(time): # tests if beginning_index is included in time
            self.v_line.setPos(time[self.beginning_index])  # starts at the first gate point, in ms
        else: # if out of bounds for time
            self.v_line.setPos(time[0]) # set position of vertical line at 0
        print("Beginning index is: "+str(self.beginning_index))
    

    def on_click(self, event):
        """
        Track mouse movement
        Function activated when plot is clicked
        """
        pos = self.datapross_plot.plotItem.vb.mapSceneToView(event.scenePos()) # get (x,y) position of mouse detected on the plot
        if self.tracking:
            self.tracking = False  # tracking locked when clicked
            self.beginning_index = int(pos.x()) # x-axis value is stored as beginning_index
        else:
            self.v_line.setPos(pos.x()) # set position to follow the mouse x-axis position
            self.tracking = True  # tracking unlocked


    def on_mouse_move(self, event):
        """
        Function activated when tracking=True and make v_line follow user mouse
        """
        if self.tracking:
            pos = self.datapross_plot.plotItem.vb.mapSceneToView(event) # get (x,y) position of mouse detected on the plot
            self.v_line.setPos(pos.x()) # set position to follow the mouse x-axis position
            self.beginning_index = int(pos.x()) # x-axis value is stored as beginning_index


    def close_and_update_main(self):  # signal function
        file_name = self.file_input.text()  # retrieve file_name from file_input
        self.emit_all_parameters()
            # synchronization with emit_all_parameters, so that both signals are complete when confirming processing
        if file_name:
            self.update_filename_signal.emit(file_name)  # emit signal to HeatmapWindow
            self.close()  # close DataProcessingWindow only if file_name specified


    def emit_all_parameters(self): # signal function
        # emit a signal when user toggles a parameter
        sampling_rate = int(self.samplrate_input.text())
        pulse_time = int(self.segtime_input.text())
        delay_time = int(self.pausetime_input.text())
        scan_list = [pulse_time, delay_time]
        selected_drains = [key for key, btn in self.drain_buttons.items() if btn.isChecked()]
        beginning_index = getattr(self, "beginning_index", 0) # get value of self with attribute "beginning_index" if existing, else 0
        self.update_all_parameters.emit(sampling_rate, scan_list, selected_drains, beginning_index) # emit signal to HeatmapWindow


    def remove_drain(self): # signal function, update when user toggles a drain button
        self.emit_all_parameters()


# =================== GUI WINDOW ===================


# MAIN WINDOW
class HeatmapWindow(QMainWindow):


    # 1. main menu window initialization
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Electrophysiology Heatmap")
        self.setGeometry(100, 100, 1000, 1600)
        central_widget = QWidget()
        self.setCentralWidget(central_widget) # create central widget because QMainWindow can not have layout directly
        self.layout = QVBoxLayout() # vertical layout
        central_widget.setLayout(self.layout)
        # 2. buttons
            # 2.1. New file button
        self.new_file_button = QPushButton("Create a new file")
        self.new_file_button.clicked.connect(self.open_datapross_window)
            # 2.2. Open folder button
        self.open_folder_button = QPushButton("Open processed data")
        self.open_folder_button.clicked.connect(self.open_folder)
        self.open_folder_button.clicked.connect(self.open_heatmap_window_foropenfolder)
            # descriptions (labels)
        self.label_new = QLabel("Choose a txt file to create a new heatmap.")
        self.label_open = QLabel("Open a folder with processed txt files, to visualize a heatmap.")
            # center text in labels
        self.label_new.setAlignment(Qt.AlignCenter)
        self.label_open.setAlignment(Qt.AlignCenter)
            # grid layout
            # qt alignment: https://www.geeksforgeeks.org/qt-alignment-in-pyqt5/
        grid = QGridLayout()
        grid.addWidget(self.label_new, 1, 0, alignment=Qt.AlignBottom)
        grid.addWidget(self.label_open, 1, 1, alignment=Qt.AlignBottom)
        grid.addWidget(self.new_file_button, 2, 0, alignment=Qt.AlignTop)
        grid.addWidget(self.open_folder_button, 2, 1, alignment=Qt.AlignTop)
        self.layout.addLayout(grid) # add elements to layout

            # menu bar
        self._createMenuBar()

        # variables  initialization
        self.dataprocessing_window = None
        self.heatmap_window = None
        self.file_name = None
        self.processed_data = None


    # =============== ACTION FUNCTIONS ===============
    # 2.1.1. data processing window opening
    def open_datapross_window(self):
        if self.dataprocessing_window is None or not self.dataprocessing_window.isVisible():
            self.dataprocessing_window = DataProcessingWindow()
            self.dataprocessing_window.update_filename_signal.connect(self.open_heatmap_window_fornewfile)  # connect filename signal
            self.dataprocessing_window.update_all_parameters.connect(self.update_parameters) # connect parameters signal
            self.dataprocessing_window.show()


    # 2.1.3. store parameters from DataProcessingWindow
    def update_parameters(self, sampling_rate, scan_list, selected_drains, beginning_index):
        # put variable as instance variable
        self.sampling_rate = sampling_rate
        self.pulse_time = scan_list[0]
        self.delay_time = scan_list[1]
        self.selected_drains = selected_drains
        self.beginning_index = beginning_index


    # 2.1.4. heatmap window initialization for new file
    def open_heatmap_window_fornewfile(self, file_name):
        self.file_name = file_name # as instance variable
        # clear previous layout and widgets
        while self.layout.count():
            item = self.layout.takeAt(0) # remove all widgets
            widget = item.widget()
            if widget is not None:
                widget.deleteLater() # force deletion
        # function to create new options (widgets)
        self._create_heatmap_widgets()
        self.repaint() # force UI refresh
        self.update()
    

    # 2.2.2. heatmap window initialization for open folder
    def open_heatmap_window_foropenfolder(self):
        # clear previous layout and widgets
        while self.layout.count():
            item = self.layout.takeAt(0) # remove all widgets
            widget = item.widget()
            if widget is not None:
                widget.deleteLater() # force deletion
        # function to create new options (widgets)
        self._create_heatmap_widgets()
        self.repaint() # force UI refresh
        self.update()
    

    # main menu bar
    def _createMenuBar(self):
        menuBar = self.menuBar() # menu bar of the QMainWindow
        # File Menu
        fileMenu = menuBar.addMenu("&File")
        # 'New File' action
        self.newAction = QAction("New File", self)
        self.newAction.triggered.connect(self.new_file)
        fileMenu.addAction(self.newAction)
        # 'Open Folder' action, selects directory
        self.openAction = QAction("Open Folder", self)
        self.openAction.triggered.connect(self.open_folder)
        fileMenu.addAction(self.openAction)
        # 'Exit' action
        self.exitAction = QAction("Exit", self)
        self.exitAction.triggered.connect(self.close)
        fileMenu.addAction(self.exitAction)


    # 2.1. action for 'New File'
    def new_file(self):
        self.open_datapross_window()


    # 2.2.1.1. needed in open_folder function below
    def load_metadata(self, folder_path):
        metadata_path = os.path.join(folder_path, "metadata.txt")
        if os.path.exists(metadata_path): # check if metadata.txt exists
            with open(metadata_path, "r") as f:
                for line in f:
                    key, value = line.strip().split("=")
                    setattr(self, key, float(value)) # convert values to float
            print("Metadata loaded:", self.sampling_rate, self.pulse_time, self.delay_time, self.beginning_index)
        else:
            print("Warning: No metadata file found. Default values may be missing.")


    # 2.2.1. action for 'Open Folder'
    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        
        self.open_heatmap_window_foropenfolder() # updates HeatmapWindow
        if folder_path:
            self.status_label.setText(f"Status: Opened folder {folder_path}")
            self.load_metadata(folder_path) # see 2.2.1.1
            # list all .txt files in the folder, except metadata.txt
            txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt') and f!="metadata.txt"]
            if txt_files:
                self.status_label.setText(f"Status: Found {len(txt_files)} .txt files in {folder_path}")
            else:
                self.status_label.setText(f"Status: No .txt files found in {folder_path}")
            # dictionnary to store processed data
            self.processed_data = {}
            for file in txt_files:
                file_path = os.path.join(folder_path, file)
                col_name = os.path.splitext(file)[0] # column name from filename
                try:
                    df = pd.read_csv(file_path, delimiter="\t")  # load as DataFrame
                    self.processed_data[col_name] = df  # processed_data dictionary
                except Exception as e:
                    self.status_label.setText(f"Status: Error loading {file}: {str(e)}")


        self.status_label.setText(f"Status: Successfully loaded {len(self.processed_data)} files.") # update status_label with success message



    # =============== HEATMAP INTERFACE WINDOW ===============
    # 3. window for heatmap
    def _create_heatmap_widgets(self):
        if self.file_name:
            self.load_file() # see 4., activate load_file only if file_name has an input (by 2.1. New file button)
        # main window
        self.setWindowTitle("Electrophysiology Heatmap")

        # status label
        self.status_label = QLabel("Status: Waiting for input...")
        self.layout.addWidget(self.status_label)
        # threshold label
        self.threshold_label = QLabel("Threshold Factor: 50")
        self.layout.addWidget(self.threshold_label)
        # threshold slider
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(1)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        self.layout.addWidget(self.threshold_slider)
        # horizontal layout
        heatmap_button_layout = QHBoxLayout()
        # heatmap button generate
        self.generate_heatmap_button = QPushButton("Generate Heatmap")
        self.generate_heatmap_button.clicked.connect(self.generate_heatmap)
        heatmap_button_layout.addWidget(self.generate_heatmap_button)
        # dropdown for heatmap type selection
        self.heatmap_selection = QComboBox()
        self.heatmap_selection.addItems(["Amplitude Peak to Peak", "Sign", "Frequency","Propagation"])  # options
        self.heatmap_selection.currentIndexChanged.connect(self.generate_heatmap)  # update heatmap when selected
        heatmap_button_layout.addWidget(self.heatmap_selection)
        self.layout.addLayout(heatmap_button_layout)

        # heatmap widget
        plot_item = pg.PlotItem()
        plot_item.hideAxis('left') # hide y-axis
        plot_item.hideAxis('bottom') # hide x-axis
        self.heatmap_widget = pg.ImageView(view=plot_item) # displays 2D image data
        self.heatmap_widget.ui.roiBtn.hide() # hide pyqtgraph built-in ROI button
        self.heatmap_widget.ui.menuBtn.hide() # hide pyqtgraph built-in Menu button
        self.layout.addWidget(self.heatmap_widget)

        # horizontal separator
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.HLine)
        self.separator.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(self.separator)
        # 6. exportation button
        self.export_button = QPushButton("Export Heatmap")
        self.export_button.clicked.connect(self.export_heatmap)
        self.layout.addWidget(self.export_button)
        # 7. heatmap event click event
        self.heatmap_widget.scene.sigMouseClicked.connect(self.on_heatmap_click)
        # 5. heatmap creation
        self.generate_heatmap()
    


    # =============== ACTION FUNCTION ===============
    def update_threshold_label(self, value): # function to retrieve factor threshold label
        self.threshold_label.setText(f"Threshold Factor: {value}")


    # =============== DATA IMPORTATION ===============
    # used in load_file below, save parameters of DataProcessingWindow to metadata.txt
    def save_metadata(self, folder_path):
        metadata_path = os.path.join(folder_path, "metadata.txt")
        with open(metadata_path, "w") as f:
            f.write(f"sampling_rate={self.sampling_rate}\n")
            f.write(f"pulse_time={self.pulse_time}\n")
            f.write(f"delay_time={self.delay_time}\n")
            f.write(f"beginning_index={self.beginning_index}\n")


    # 4. load_file function, from a new txt file (2.1. New file button)
    def load_file(self):
        # file name from the input field
        file_name = self.file_name
        if not file_name:
            self.status_label.setText("Status: Please specify a valid file path.")
            return
        
        try:
            # window for asking user the output directory file
            output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
            if not output_dir:  # if the user cancels the directory selection
                self.status_label.setText("Status: No output directory selected.")
                return
            os.makedirs(output_dir, exist_ok=True) # test if exist or not, create a file if not exist
            
            df = pd.read_csv(file_name, delimiter="\t", encoding="iso-8859-1", skiprows=2, low_memory=False, dtype=str)

            # data cleaning
            df.drop(df.columns[[0]], axis=1, inplace=True) # drop time column
            if hasattr(self, "selected_drains") and self.selected_drains: # hasttr: returns boolean whether object has attribute "selected_drains" or not
                drain_columns = [int(drain) - 1 for drain in self.selected_drains] # convert drain numbers to column indices
                df.drop(df.columns[drain_columns], axis=1, inplace=True) # drop unwanted columns from DataProcessingWindow
            df.columns = df.columns.str.strip() # remove spaces from column name
            df = df.apply(pd.to_numeric, errors='coerce')
                # parameter
            lowcut = 500 # in Hz, for highpass, to straighten the signal

            # activate processing
            self.processed_data = process_all_columns(df, output_dir, self.sampling_rate, self.pulse_time, self.delay_time, lowcut, self.beginning_index)
            self.save_metadata(output_dir) # see above

        except Exception as e:
            self.status_label.setText(f"Status: Error - {e}")
            return
    

    # =============== HEATMAP GENERATION FUNCTIONS ===============


    # 5.1. function to generate heatmap
    def generate_heatmap(self):
        # removes text if present
        if hasattr(self, "textItems") and self.textItems: # hasattr: returns boolean whether object has attribute "textItems" or not
            for item in self.textItems:
                self.heatmap_widget.removeItem(item)
            self.textItems.clear()  # reset the list of text items
        if not self.processed_data: # if the file has not been created yet
            self.status_label.setText("Status: No processed data available.")
            return
        heatmaps = generate_heatmap_data(self, self.processed_data, self.threshold_slider.value()) # dictionnary of values: Amplitude Peak to Peak, Sign, Frequency, Propagation
        selected_type = self.heatmap_selection.currentText()  # selected heatmap type from dropdown menu
        heatmap_data = heatmaps[selected_type]  # corresponding heatmap
        self.status_label.setText("Status: Heatmap Generated!")
        # small font for heatmap text
        small_font = QFont("Arial", 5)
        # values for heatmap_data
        min_val = heatmap_data.min().min() # set minimum colormap color to minimum value of each heatmap
        max_val = heatmap_data.max().max() # set maximum colormap color to maximum value of each heatmap
        # colormap
        colormap = pg.colormap.get("turbo") # blue (low amplitude) to red (high amplitude), with yellow for neutral
        if selected_type == "Sign":
            colormap = pg.colormap.get("CET-D1") # blue (negative) to red (positive), with white for neutral

        # initialization of list to track text objects
        self.textItems = []

        # initialization of heatmap
        self.heatmap_widget.setImage(heatmap_data.to_numpy().T, levels=(min_val, max_val))
            # transposition T because setImage assumes (column, row) data, whereas this is (row,column)
            # https://pyqtgraph.readthedocs.io/en/latest/api_reference/widgets/imageview.html
        self.heatmap_widget.setColorMap(colormap)
        self.heatmap_widget.view.setRange(xRange=[-1, heatmap_data.shape[0]+1], yRange=[-1, heatmap_data.shape[1]+1])
            # unzoom the heatmap so labels can be seen
        # heatmap legend
        drain_legend = pg.TextItem("Drain lines", color=(255,255,255), anchor=(0.5,0.5), angle=90) # text
        drain_legend.setFont(QFont("Arial", 12)) # font
        drain_legend.setPos(heatmap_data.shape[1]+1.5, heatmap_data.shape[0]/2) # position
        self.heatmap_widget.addItem(drain_legend) # add legend to widget
        self.textItems.append(drain_legend) # add to textItems list
        gate_legend = pg.TextItem("Gate lines", color=(255,255,255), anchor=(0.5,0.5))
        gate_legend.setFont(QFont("Arial", 12))
        gate_legend.setPos(heatmap_data.shape[1]/2, heatmap_data.shape[0]+1.5)
        self.heatmap_widget.addItem(gate_legend)
        self.textItems.append(gate_legend)
        # amplitude peak to peak heatmap
        if selected_type == "Amplitude Peak to Peak":
            # heatmap title
            title = pg.TextItem("Peak to peak amplitude (µV)", color=(255,255,255), anchor=(0.5,0.5))
            title.setFont(QFont("Arial", 14))
            title.setPos(14.5, -1)
            self.heatmap_widget.addItem(title)
            self.textItems.append(title)
            # unity
            unity_legend = pg.TextItem("µV", color=(255,255,255), anchor=(0.5,0.5))
            unity_legend.setFont(QFont("Arial", 12))
            unity_legend.setPos(30, 0.5)
            self.heatmap_widget.addItem(unity_legend)
            self.textItems.append(unity_legend)
            # labels for each square
            for i in range(heatmap_data.shape[0]):  # iterate over rows, y-axis
                # drain lines label
                drain_label = pg.TextItem(str(i+1), color=(255,255,255), anchor=(0.5, 0.5))
                drain_label.setPos(heatmap_data.shape[1] + 0.5, i + 0.5)
                self.heatmap_widget.addItem(drain_label)
                self.textItems.append(drain_label)
                for j in range(heatmap_data.shape[1]):  # iterate over columns, x-axis
                    # gate lines label
                    gate_label = pg.TextItem(str(j+1), color=(255,255,255), anchor=(0.5, 0.5))
                    gate_label.setPos(j + 0.5, heatmap_data.shape[0] + 0.5)
                    self.heatmap_widget.addItem(gate_label)
                    self.textItems.append(gate_label)
                    # retrieve the numerical value to display
                    value = heatmap_data.iloc[i, j]
                    if not np.isnan(value):
                        # change text color depending on square color (value)
                        text_color = (0,0,0) if ( value > max_val/4 and value < 3*max_val/4 ) else (255,255,255)
                        text_value = pg.TextItem(str(round(value, 1)), color=text_color, anchor=(0.5, 0.5))
                        text_value.setFont(small_font)
                        text_value.setPos(j + 0.5, i + 0.5)
                        self.heatmap_widget.addItem(text_value)
                        self.textItems.append(text_value)
        # sign value heatmap
        if selected_type == "Sign":
            # heatmap title
            title = pg.TextItem("Sign (µV)", color=(255,255,255), anchor=(0.5,0.5))
            title.setFont(QFont("Arial", 14))
            title.setPos(14.5, -1)
            self.heatmap_widget.addItem(title)
            self.textItems.append(title)
            # unity
            unity_legend = pg.TextItem("µV", color=(255,255,255), anchor=(0.5,0.5))
            unity_legend.setFont(QFont("Arial", 12))
            unity_legend.setPos(30, 0.5)
            self.heatmap_widget.addItem(unity_legend)
            self.textItems.append(unity_legend)
            # labels for each square
            for i in range(heatmap_data.shape[0]):  # iterate over rows, y-axis
                # drain lines label
                drain_label = pg.TextItem(str(i+1), color=(255,255,255), anchor=(0.5, 0.5))
                drain_label.setPos(heatmap_data.shape[1] + 0.5, i + 0.5)
                self.heatmap_widget.addItem(drain_label)
                self.textItems.append(drain_label)
                for j in range(heatmap_data.shape[1]):  # iterate over columns, x-axis
                    # gate lines label
                    gate_label = pg.TextItem(str(j+1), color=(255,255,255), anchor=(0.5, 0.5))
                    gate_label.setPos(j + 0.5, heatmap_data.shape[0] + 0.5)
                    self.heatmap_widget.addItem(gate_label)
                    self.textItems.append(gate_label)
                    # retrieve the numerical value to display
                    value = heatmap_data.iloc[i, j]
                    if not np.isnan(value):
                        text_color = (0,0,0)
                        text_value = pg.TextItem(str(round(value, 1)), color=text_color, anchor=(0.5, 0.5))
                        text_value.setFont(small_font)
                        text_value.setPos(j + 0.5, i + 0.5)
                        self.heatmap_widget.addItem(text_value)
                        self.textItems.append(text_value)
        # frequency heatmap
        if selected_type == "Frequency":
            # heatmap titles
            title = pg.TextItem("Frequency (Hz)", color=(255,255,255), anchor=(0.5,0.5))
            title.setFont(QFont("Arial", 14))
            title.setPos(14.5, -1)
            self.heatmap_widget.addItem(title)
            self.textItems.append(title)
            # unity
            unity_legend = pg.TextItem("Hz", color=(255,255,255), anchor=(0.5,0.5))
            unity_legend.setFont(QFont("Arial", 12))
            unity_legend.setPos(30, 0.5)
            self.heatmap_widget.addItem(unity_legend)
            self.textItems.append(unity_legend)
            # labels for each square
            for i in range(heatmap_data.shape[0]):  # iterate over rows, y-axis
                # drain lines label
                drain_label = pg.TextItem(str(i+1), color=(255,255,255), anchor=(0.5, 0.5))
                drain_label.setPos(heatmap_data.shape[1] + 0.5, i + 0.5)
                self.heatmap_widget.addItem(drain_label)
                self.textItems.append(drain_label)
                for j in range(heatmap_data.shape[1]):  # iterate over columns, x-axis
                    # gate lines label
                    gate_label = pg.TextItem(str(j+1), color=(255,255,255), anchor=(0.5, 0.5))
                    gate_label.setPos(j + 0.5, heatmap_data.shape[0] + 0.5)
                    self.heatmap_widget.addItem(gate_label)
                    self.textItems.append(gate_label)
                    # retrieve the numerical value to display
                    value = heatmap_data.iloc[i, j]
                    if not np.isnan(value):
                        # change text color depending on square color (value)
                        text_color = (0,0,0) if ( value > max_val/4 and value < 3*max_val/4 ) else (255,255,255)
                        text_value = pg.TextItem(str(round(value, 2)), color=text_color, anchor=(0.5, 0.5))
                        text_value.setFont(small_font)
                        text_value.setPos(j + 0.5, i + 0.5)
                        self.heatmap_widget.addItem(text_value)
                        self.textItems.append(text_value)
        # propagation heatmap
        if selected_type == "Propagation":
            # heatmap title
            title = pg.TextItem("Propagation: first peak time (ms)", color=(255,255,255), anchor=(0.5,0.5))
            title.setFont(QFont("Arial", 14))
            title.setPos(14.5, -1)
            self.heatmap_widget.addItem(title)
            self.textItems.append(title)
            # unity
            unity_legend = pg.TextItem("ms", color=(255,255,255), anchor=(0.5,0.5))
            unity_legend.setFont(QFont("Arial", 12))
            unity_legend.setPos(30, 0.5)
            self.heatmap_widget.addItem(unity_legend)
            self.textItems.append(unity_legend)
            # labels for each square
            for i in range(heatmap_data.shape[0]):  # iterate over rows, y-axis
                # drain lines label
                drain_label = pg.TextItem(str(i+1), color=(255,255,255), anchor=(0.5, 0.5))
                drain_label.setPos(heatmap_data.shape[1] + 0.5, i + 0.5)
                self.heatmap_widget.addItem(drain_label)
                self.textItems.append(drain_label)
                for j in range(heatmap_data.shape[1]):  # iterate over columns, x-axis
                    # gate lines label
                    gate_label = pg.TextItem(str(j+1), color=(255,255,255), anchor=(0.5, 0.5))
                    gate_label.setPos(j + 0.5, heatmap_data.shape[0] + 0.5)
                    self.heatmap_widget.addItem(gate_label)
                    self.textItems.append(gate_label)
                    # retrieve the numerical value to display
                    value = heatmap_data.iloc[i, j]
                    if not np.isnan(value):
                        # change text color depending on square color (value)
                        text_color = (0,0,0) if ( value > max_val/4 and value < 3*max_val/4 ) else (255,255,255)
                        text_value = pg.TextItem(str(round(value, 1)), color=text_color, anchor=(0.5, 0.5))
                        text_value.setFont(small_font)
                        text_value.setPos(j + 0.5, i + 0.5)
                        self.heatmap_widget.addItem(text_value)
                        self.textItems.append(text_value)


    # =============== DATA EXPORTATION ===============
    # 6.1. data exportation function
    def export_heatmap(self):
        if not self.processed_data:
            QMessageBox.warning(self, "Export Error", "No heatmap available to export.")
            return
        # opens a window to choose repertory to save heatmap image
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Heatmap",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )
        if file_name:
            #  file format
            if file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg"):
                format = "JPEG"
            else:
                if not file_name.lower().endswith(".png"):
                    file_name += ".png"
                format = "PNG"


            # scale factor (increase for higher quality)
            scale_factor = 3  # or 2, 3, or 4 for better quality
            # widget's bounding rectangle size
            width = self.heatmap_widget.width()
            height = self.heatmap_widget.height()
            # high-resolution QImage
            high_res_image = QImage(
                width * scale_factor, height * scale_factor, QImage.Format_ARGB32
            )
            # render at high resolution
            painter = QPainter(high_res_image)
            painter.setRenderHint(QPainter.Antialiasing)  # for smoother text & graphics
            painter.scale(scale_factor, scale_factor)  # scale rendering
            self.heatmap_widget.render(painter)  # render the ImageView to the image
            painter.end()
            # save the high-res image
            high_res_image.save(file_name, format, quality=100)  # JPEG quality = 100


            self.status_label.setText(f"Status: Heatmap saved as {file_name}")


    # =============== ELECTRODE SIGNAL PLOTTING ===============
    def on_heatmap_click(self, event):
        """
        Function to plot the signal of the clicked electrode on the heatmap.
        """
        if not self.processed_data: # nothing happens if no data is opened
            return
        # mouse coordinate detection
        pos = event.scenePos()
        mouse_point = self.heatmap_widget.getImageItem().mapFromScene(pos)
        x, y = int(mouse_point.x()), int(mouse_point.y())
        # heatmap limits test
        if x < 0 or x >= 28 or y < 0 or y >= len(self.processed_data):
            return
        # electrode detection
        source_line = list(self.processed_data.keys())[y]  # drain source line
        electrode_name = f"Electrode_{x + 1}"  # column (electrode)
        if electrode_name not in self.processed_data[source_line].columns: # if electrode x+1 not in data "DrainLine_Fi 0y"
            return
        signal_data = self.processed_data[source_line][electrode_name].dropna().values # drop NaN values
        time = np.arange(0, self.pulse_time-500-750, 1/self.sampling_rate*1000) # self.pulse_time-500-750, 500ms endcut, 750ms starting offset
        # peak detection
        threshold_factor = self.threshold_slider.value() # get threshold slider value
        threshold_activity = 0.1 * threshold_factor * np.mean(np.abs(signal_data)) # threshold to detect activity peaks, same as in generate_heatmap_data
        posi_peaks, _ = find_peaks(signal_data, height=threshold_activity, distance=int(100*self.sampling_rate/1000))
        nega_peaks, _ = find_peaks(-signal_data, height=threshold_activity, distance=int(100*self.sampling_rate/1000))
        # electrode signal plotting
        plt.figure(figsize=(10, 4))
        plt.plot(time, signal_data, label='Signal')
        plt.plot(time[posi_peaks], signal_data[posi_peaks], "x", label='Positive Peaks')
        plt.plot(time[nega_peaks], signal_data[nega_peaks], "o", label='Negative Peaks')
        plt.title(f"Signal of {electrode_name} - Line {y+1}")
        plt.xlabel(f"Time (ms)")
        plt.ylabel("Amplitude (µV)")
        plt.legend()
        plt.show()


# =================== RUN APPLICATION ===================
if __name__ == "__main__":
   app = QApplication(sys.argv)
   main_window = HeatmapWindow()
   main_window.show()
   sys.exit(app.exec_())


# =================== END OF FILE ===================

