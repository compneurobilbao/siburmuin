#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on septiembre 27, 2024, at 12:32
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, iohub, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from GLOBAL_VARIABLES_AND_FUNCTIONS
# IMPORTS
from psychopy import core
import random

#GLOBAL VARIABLES

noise_dots = 25000
grating_size = (0.5,0.5)

# Staircase test params
n_reversals_to_average = 4
stop_reversals = 5
staircase_noise_duration = 0.5

# Test params
stim_time = 2
response_time = 1.5 # time after stimuli disapears to answer


# FUNCTIONS

def show_noise(dots_white, dots_black, duration):
    duration = 0.5
    # Habilitar los puntos de ruido
    dots_white.setAutoDraw(True)
    dots_black.setAutoDraw(True)

    noise_timer = core.Clock()
    noise_timer.reset()
    
    # Mostrar el ruido durante el tiempo de duración especificado
    while noise_timer.getTime() < duration:
        win.flip()  # Actualiza la ventana en cada frame para mantener la animación
    
    # Desactivar los puntos de ruido
    dots_white.setAutoDraw(False)
    dots_black.setAutoDraw(False)

def get_random_orientation():
    return random.choice([45, 135])

def get_threshold(test_var_name : str, results_csv_path):
    data = pd.read_csv(results_csv_path)

    # Filtrar filas con reversals
    reversal_data = data[data['reversals'] > 0]

    threshold = reversal_data[test_var_name].tail(n_reversals_to_average).mean()

    return threshold

def load_sf():
    archivo_sf = './data/sf_staircase_test_data.csv'
    try:
        test_sf = get_threshold('spatial_frequency', archivo_sf)
        print(f'Se ha cargado la frecuencia espacial testada con un valor de {test_sf}')
        return test_sf

    except FileNotFoundError:
        print(f'Error: El archivo {archivo_sf} no fue encontrado.')
        return -1
        
    except Exception as e:
        print(f'Error al cargar la frecuencia espacial del archivo {archivo_sf}: {str(e)}')
        return -1
# Run 'Before Experiment' code from DATA_MANAGEMENT
import json

# TEMPORAL --> Se debe cargar de memoria o inicializar con valores nulos
threshold_values = {
    'spatial_frequency_threshold': 53.98,   # Flotante
    'flicker_threshold': 40.0,              # Flotante
    'contrast_threshold': 0.002,            # Flotante
    'color_threshold': {                    # Diccionario para colores con valores flotantes
        'red': 0.93,
        'green': 3.44
    }                 
}
'''
threshold_values = {
    'spatial_frequency_threshold': None,  # Flotante
    'flicker_threshold': None,            # Flotante
    'contrast_threshold': None,           # Flotante
    'color_threshold': {}                 # Diccionario para colores con valores flotantes
}
'''
''' MODIFICAR EL DICCIONARIO
threshold_dict['spatial_frequency_threshold'] = spatial_frequency
threshold_dict['flicker_threshold'] = flicker
threshold_dict['contrast_threshold'] = contrast
threshold_dict['color_threshold'][color_name] = color_value
'''
# Función para mostrar el diccionario completo al final del test
def display_thresholds(threshold_dict):
    print("Valores de Umbrales del Paciente:")
    print(f"Frecuencia Espacial: {threshold_dict['spatial_frequency_threshold']}")
    print(f"Flicker: {threshold_dict['flicker_threshold']}")
    print(f"Contraste: {threshold_dict['contrast_threshold']}")
    print("Umbrales de Color:")
    for color, value in threshold_dict['color_threshold'].items():
        print(f"{color}: {value}")

# Función para guardar el diccionario en un archivo JSON
def save_thresholds_to_json(threshold_dict, filename='./data/thresholds.json'):
    with open(filename, 'w') as f:
        json.dump(threshold_dict, f, indent=4)  # indent para que el JSON sea legible
    print(f"Diccionario guardado en {filename}")

# Función para cargar el diccionario desde un archivo JSON
def load_thresholds_from_json(filename='./data/thresholds.json'):
    with open(filename, 'r') as f:
        threshold_dict = json.load(f)
    print(f"Diccionario cargado desde {filename}")
    return threshold_dict

# Run 'Before Experiment' code from code_14
import pandas as pd

opacidad = 1

# Run 'Before Experiment' code from gabor_generator_2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def save_gabor_patch_image(frequency, size, c1, c2):
    amp, f = generate_gabor_patch(frequency, size)
    
    # Convertir colores a numpy arrays y expandir dimensiones para el canal de transparencia
    c1 = np.array(c1)
    c2 = np.array(c2)
    
    # Calcular los valores de color para el parche
    im_rgb_vals = (c1 * amp[:, :, None]) + (c2 * (1 - amp[:, :, None]))
    
    # Crear el canal de alfa (transparencia): 1 donde hay el parche, 0 en el fondo
    alpha_channel = f
    
    # Combinar los valores RGB con el canal alfa para crear una imagen RGBA
    im_rgba_vals = np.dstack((im_rgb_vals, alpha_channel))
    
    # Convertir a imagen
    im = Image.fromarray((im_rgba_vals * 255).astype('uint8'), 'RGBA')
    im.save(f"./images/custom_stim.png")

def generate_gabor_patch(frequency, size):
    im_range = np.arange(size)
    x, y = np.meshgrid(im_range, im_range)
    dx = x - size // 2
    dy = y - size // 2
    t = np.arctan2(dy, dx)
    r = np.sqrt(dx ** 2 + dy ** 2)
    x = r * np.cos(t)
    y = r * np.sin(t)
    
    # Transición brusca para los colores (líneas) en el patrón Gabor
    amp = np.where(np.cos(2 * np.pi * (x * frequency)) >= 0, 1, 0)
    f = np.where(r <= size // 2, 1, 0)
    
    return amp, f
def hsv_a_rgb(h, s, v):
    """
    Convierte un color desde HSV a RGB.

    Parámetros:
    h (float): Matiz (Hue) en grados (0-360).
    s (float): Saturación (Saturation) como porcentaje (0-100).
    v (float): Valor (Value) como porcentaje (0-100).

    Retorna:
    tuple: Una tupla con valores (R, G, B), cada uno en el rango de 0 a 255.
    """
    h = h % 360
    s /= 100
    v /= 100

    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    r = (r + m) * 255
    g = (g + m) * 255
    b = (b + m) * 255

    return int(round(r)), int(round(g)), int(round(b))


def normalizar_rgb(rgb):
    """
    Normaliza una tupla de valores RGB dividiendo cada componente por 255.

    Parámetros:
    rgb (tuple): Una tupla con valores (R, G, B), cada uno en el rango de 0 a 255.

    Retorna:
    tuple: Una tupla con valores normalizados (R, G, B), cada uno en el rango de 0 a 1.
    """
    return tuple(component / 255 for component in rgb)
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'magnocellular_stimuli'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\akoun\\Desktop\\Biocruces\\siburmuin\\src\\begiBrainDemo\\begiBrainDemo_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1920, 1080], fullscr=False, screen=1,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = True
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup eyetracking
    ioConfig['eyetracker.hw.mouse.EyeTracker'] = {
        'name': 'tracker',
        'controls': {
            'move': [],
            'blink':('MIDDLE_BUTTON',),
            'saccade_threshold': 0.5,
        }
    }
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = ioServer.getDevice('tracker')
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "load_config" ---
    # Run 'Begin Experiment' code from code_4
    periphereal_region_diameter = 0
    
    ################################
    ## CONFIGURACION MODIFICABLE: ##
    ################################
    nombre_pantalla = 'pantalla4'
    distancia_eyetracker = 0.65 # m
    alpha = angulo_region_central = 9 # º DEG
    periphereal_region_result = visual.ShapeStim(
        win=win, name='periphereal_region_result',
        size=[1.0, 1.0], vertices='circle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=2.0,     colorSpace='rgb',  lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    key_resp_4 = keyboard.Keyboard()
    logs2 = visual.TextStim(win=win, name='logs2',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    # Run 'Begin Experiment' code from GLOBAL_VARIABLES_AND_FUNCTIONS
    stop_reversals = 5
    FPS_logs = visual.TextStim(win=win, name='FPS_logs',
        text=None,
        font='Open Sans',
        pos=(0.35, 0.35), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    
    # --- Initialize components for Routine "LOAD_THRESHOLDS" ---
    
    # --- Initialize components for Routine "BL_1_SPATIAL_FREQ" ---
    dots_black_5 = visual.DotStim(
        win=win, name='dots_black_5',
        nDots=noise_dots, dotSize=2.0,
        speed=0.1, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=[1.75,1], fieldAnchor='center', fieldShape='square',
        signalDots='same', noiseDots='direction',dotLife=3.0,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None,
        depth=0.0)
    dots_white_5 = visual.DotStim(
        win=win, name='dots_white_5',
        nDots=noise_dots, dotSize=2.0,
        speed=0.1, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=[1.75,1], fieldAnchor='center', fieldShape='square',
        signalDots='same', noiseDots='direction',dotLife=3.0,
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None,
        depth=-1.0)
    stim = visual.GratingStim(
        win=win, name='stim',
        tex='sqr', mask='circle', anchor='center',
        ori=0.0, pos=[0,0], size=1.0, sf=1.0, phase=0.5,
        color='white', colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=512.0, interpolate=True, depth=-2.0)
    # Run 'Begin Experiment' code from code
    from psychopy.iohub import launchHubServer
    
    io = launchHubServer()
    mouse = io.devices.mouse
    
    posicion_estimulo = (0,0)
    stim_x = 0
    stim_y = 0
    
    foveal_region_pos = [0,0]
    
    #other
    gaze_position = mouse.getPosition()
    key_resp = keyboard.Keyboard()
    logs_background = visual.Rect(
        win=win, name='logs_background',
        width=(1, 0.3)[0], height=(1, 0.3)[1],
        ori=0.0, pos=(-0.5, 0.5), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    logs_background_2 = visual.Rect(
        win=win, name='logs_background_2',
        width=(0.5, 1)[0], height=(0.5, 1)[1],
        ori=0.0, pos=(0.75, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-6.0, interpolate=True)
    logs = visual.TextStim(win=win, name='logs',
        text=None,
        font='Open Sans',
        pos=(-0.45, 0.45), height=0.035, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    logs_parametros_trial = visual.TextStim(win=win, name='logs_parametros_trial',
        text=None,
        font='Open Sans',
        pos=(0.5, 0), height=0.025, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    logs_coordenadas_mirada = visual.TextStim(win=win, name='logs_coordenadas_mirada',
        text=None,
        font='Open Sans',
        pos=(-0.45, -0.45), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    gaze = visual.ShapeStim(
        win=win, name='gaze',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
        opacity=0.4, depth=-10.0, interpolate=True)
    
    # --- Initialize components for Routine "BL_2_COLOR" ---
    # Run 'Begin Experiment' code from code_14
    from psychopy.iohub import launchHubServer
    
    io = launchHubServer()
    mouse = io.devices.mouse
    
    posicion_estimulo = (0,0)
    stim_x = 0
    stim_y = 0
    
    foveal_region_pos = [0,0]
    
    #other
    gaze_position = mouse.getPosition()
    key_resp_10 = keyboard.Keyboard()
    logs_background_9 = visual.Rect(
        win=win, name='logs_background_9',
        width=(1, 0.3)[0], height=(1, 0.3)[1],
        ori=0.0, pos=(-0.5, 0.5), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    logs_background_10 = visual.Rect(
        win=win, name='logs_background_10',
        width=(0.5, 1)[0], height=(0.5, 1)[1],
        ori=0.0, pos=(0.75, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    logs_7 = visual.TextStim(win=win, name='logs_7',
        text=None,
        font='Open Sans',
        pos=(-0.45, 0.45), height=0.035, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    logs_parametros_trial_6 = visual.TextStim(win=win, name='logs_parametros_trial_6',
        text=None,
        font='Open Sans',
        pos=(0.5, 0), height=0.025, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    logs_coordenadas_mirada_6 = visual.TextStim(win=win, name='logs_coordenadas_mirada_6',
        text=None,
        font='Open Sans',
        pos=(-0.45, -0.45), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    gaze_6 = visual.ShapeStim(
        win=win, name='gaze_6',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
        opacity=0.4, depth=-8.0, interpolate=True)
    stim_img = visual.ImageStim(
        win=win,
        name='stim_img', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=grating_size,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=512.0, interpolate=True, depth=-9.0)
    dots_white_6 = visual.DotStim(
        win=win, name='dots_white_6',
        nDots=noise_dots, dotSize=2.0,
        speed=0.1, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=[1.75,1], fieldAnchor='center', fieldShape='square',
        signalDots='same', noiseDots='direction',dotLife=3.0,
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None,
        depth=-10.0)
    dots_black_6 = visual.DotStim(
        win=win, name='dots_black_6',
        nDots=noise_dots, dotSize=2.0,
        speed=0.1, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=[1.75,1], fieldAnchor='center', fieldShape='square',
        signalDots='same', noiseDots='direction',dotLife=3.0,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None,
        depth=-11.0)
    
    # --- Initialize components for Routine "BL_3_CONTRAST" ---
    stim_5 = visual.GratingStim(
        win=win, name='stim_5',
        tex='sqr', mask='circle', anchor='center',
        ori=0.0, pos=[0,0], size=1.0, sf=1.0, phase=0.5,
        color='white', colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=512.0, interpolate=True, depth=0.0)
    # Run 'Begin Experiment' code from code_8
    from psychopy.iohub import launchHubServer
    
    io = launchHubServer()
    mouse = io.devices.mouse
    
    posicion_estimulo = (0,0)
    stim_x = 0
    stim_y = 0
    
    foveal_region_pos = [0,0]
    
    #other
    gaze_position = mouse.getPosition()
    key_resp_9 = keyboard.Keyboard()
    logs_background_7 = visual.Rect(
        win=win, name='logs_background_7',
        width=(1, 0.3)[0], height=(1, 0.3)[1],
        ori=0.0, pos=(-0.5, 0.5), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    logs_background_8 = visual.Rect(
        win=win, name='logs_background_8',
        width=(0.5, 1)[0], height=(0.5, 1)[1],
        ori=0.0, pos=(0.75, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    logs_6 = visual.TextStim(win=win, name='logs_6',
        text=None,
        font='Open Sans',
        pos=(-0.45, 0.45), height=0.035, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    logs_parametros_trial_5 = visual.TextStim(win=win, name='logs_parametros_trial_5',
        text=None,
        font='Open Sans',
        pos=(0.5, 0), height=0.025, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    logs_coordenadas_mirada_5 = visual.TextStim(win=win, name='logs_coordenadas_mirada_5',
        text=None,
        font='Open Sans',
        pos=(-0.45, -0.45), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    gaze_5 = visual.ShapeStim(
        win=win, name='gaze_5',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
        opacity=0.4, depth=-8.0, interpolate=True)
    dots_white_7 = visual.DotStim(
        win=win, name='dots_white_7',
        nDots=noise_dots, dotSize=2.0,
        speed=0.1, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=[1.75,1], fieldAnchor='center', fieldShape='square',
        signalDots='same', noiseDots='direction',dotLife=3.0,
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None,
        depth=-9.0)
    dots_black_7 = visual.DotStim(
        win=win, name='dots_black_7',
        nDots=noise_dots, dotSize=2.0,
        speed=0.1, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=[1.75,1], fieldAnchor='center', fieldShape='square',
        signalDots='same', noiseDots='direction',dotLife=3.0,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None,
        depth=-10.0)
    
    # --- Initialize components for Routine "BL_4_TEMPORAL_FREQ" ---
    # Run 'Begin Experiment' code from code_15
    from psychopy.iohub import launchHubServer
    
    io = launchHubServer()
    mouse = io.devices.mouse
    
    posicion_estimulo = (0,0)
    stim_x = 0
    stim_y = 0
    
    foveal_region_pos = [0,0]
    
    #other
    gaze_position = mouse.getPosition()
    key_resp_11 = keyboard.Keyboard()
    logs_background_11 = visual.Rect(
        win=win, name='logs_background_11',
        width=(1, 0.3)[0], height=(1, 0.3)[1],
        ori=0.0, pos=(-0.5, 0.5), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    logs_background_12 = visual.Rect(
        win=win, name='logs_background_12',
        width=(0.5, 1)[0], height=(0.5, 1)[1],
        ori=0.0, pos=(0.75, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    logs_8 = visual.TextStim(win=win, name='logs_8',
        text=None,
        font='Open Sans',
        pos=(-0.45, 0.45), height=0.035, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    logs_parametros_trial_7 = visual.TextStim(win=win, name='logs_parametros_trial_7',
        text=None,
        font='Open Sans',
        pos=(0.5, 0), height=0.025, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    logs_coordenadas_mirada_7 = visual.TextStim(win=win, name='logs_coordenadas_mirada_7',
        text=None,
        font='Open Sans',
        pos=(-0.45, -0.45), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    stim_7 = visual.GratingStim(
        win=win, name='stim_7',
        tex='sqr', mask='circle', anchor='center',
        ori=0.0, pos=[0,0], size=1.0, sf=1.0, phase=0.5,
        color='white', colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=512.0, interpolate=True, depth=-7.0)
    gaze_7 = visual.ShapeStim(
        win=win, name='gaze_7',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
        opacity=0.4, depth=-8.0, interpolate=True)
    dots_white_8 = visual.DotStim(
        win=win, name='dots_white_8',
        nDots=noise_dots, dotSize=2.0,
        speed=0.1, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=[1.75,1], fieldAnchor='center', fieldShape='square',
        signalDots='same', noiseDots='direction',dotLife=3.0,
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None,
        depth=-9.0)
    dots_black_8 = visual.DotStim(
        win=win, name='dots_black_8',
        nDots=noise_dots, dotSize=2.0,
        speed=0.1, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=[1.75,1], fieldAnchor='center', fieldShape='square',
        signalDots='same', noiseDots='direction',dotLife=3.0,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None,
        depth=-10.0)
    
    # --- Initialize components for Routine "BL_5_SEMANTIC_STIM" ---
    semantic_stim = visual.ImageStim(
        win=win,
        name='semantic_stim', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0.0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_resp_13 = keyboard.Keyboard()
    logs_background_13 = visual.Rect(
        win=win, name='logs_background_13',
        width=(0.5, 1)[0], height=(0.5, 1)[1],
        ori=0.0, pos=(0.75, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    logs_parametros_trial_8 = visual.TextStim(win=win, name='logs_parametros_trial_8',
        text=None,
        font='Open Sans',
        pos=(0.5, 0), height=0.025, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    right_arrow = visual.ShapeStim(
        win=win, name='right_arrow', vertices='arrow',
        size=(0.2, 0.2),
        ori=90.0, pos=(0.35, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    left_arrow = visual.ShapeStim(
        win=win, name='left_arrow', vertices='arrow',
        size=(0.2, 0.2),
        ori=-90.0, pos=(-0.35, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-6.0, interpolate=True)
    right_arrow_text = visual.TextStim(win=win, name='right_arrow_text',
        text='Animal',
        font='Open Sans',
        pos=(0.35, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    left_arrow_text = visual.TextStim(win=win, name='left_arrow_text',
        text='Inerte',
        font='Open Sans',
        pos=(-0.35, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "load_config" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('load_config.started', globalClock.getTime())
    # Run 'Begin Routine' code from code_4
    import json
    import math
    
    def calculate_diameter(excentricity, distance_to_screen, screen_height  = None):
        '''
        Calculates the diameter of circunference correspondint to the excentricity angle depending on the screen height and distance to screen.
        Params:
            -excentricity: angle of the excentricity in degrees
            -distance_to_screen: distance between patient and screen in meters
            -screen_height: height of the screen in meters (default is None, in this case the function will only return the diameter in unit that psychopy understands)
        Returns: 
            -diameter_unit: unit diameter (this is the diameter that psychopy understands, it should match with the diameter in meters when used)
            -diameter_m: diameter in meters (this is the real diameter it should have in the screen)
        '''
    
        if screen_height == None:
            diameter_m = 2 * distance_to_screen * math.sin(math.radians(excentricity))
            return None, diameter_m
        else:
            diameter_unit = (2 * distance_to_screen * math.sin(math.radians(excentricity)))/screen_height
            diameter_m = 2 * distance_to_screen * math.sin(math.radians(excentricity))
            return diameter_unit, diameter_m
    
    # Cargar el archivo JSON
    def cargar_configuracion(nombre_pantalla):
        with open('screen_config.json', 'r') as file:
            config = json.load(file)
        
        # Seleccionar la configuración específica
        pantalla_config = config.get(nombre_pantalla)
        
        if pantalla_config:
            nombre = pantalla_config['nombre']
            tamanyo_pulgadas = pantalla_config['tamanyo']
            dim_y = pantalla_config['dim_y']
            frecuencia_monitor = pantalla_config['frecuencia']
            
            print(f'Se ha cargado la siguiente configuracion:\n'
                  f'Pantalla {nombre_pantalla} de {tamanyo_pulgadas} pulgadas con altura {dim_y} m')
            return nombre, tamanyo_pulgadas, dim_y,frecuencia_monitor
        else:
            print("Configuración de pantalla no encontrada.")
            return None, None, None, None  # Devolver None para cada valor esperado
    
    nombre, tamanyo_pulgadas, dim_y, frecuencia_monitor = cargar_configuracion(nombre_pantalla)
    
    
    if nombre:  # Comprobar que nombre no es None antes de usar las variables
        # Calcular el diámetro de la frontera de la periferia
       
        diameter_unit, diameter_m = calculate_diameter(alpha, distancia_eyetracker, dim_y)
        periphereal_region_diameter = diameter_unit
    
        log = f'Se ha cargado la configuracion de la {nombre_pantalla}:\n Pantalla {nombre} de {tamanyo_pulgadas} pulgadas con altura {dim_y} m y {frecuencia_monitor} Hz.'
        print(log)
        
    logs2.setText(
                 f'Se ha cargado la configuracion de la {nombre_pantalla}:\n Pantalla {nombre} de {tamanyo_pulgadas} pulgadas con altura {dim_y} m y {frecuencia_monitor} Hz.\n' 
                 f'Para una distancia de {distancia_eyetracker} m entre el sujeto y la pantalla, el diametro debe ser de {diameter_unit:.2f} u.\n'
                 f'El diámetro equivalente es de {diameter_m:.2f} m'
                 )
    periphereal_region_result.setSize(periphereal_region_diameter)
    key_resp_4.keys = []
    key_resp_4.rt = []
    _key_resp_4_allKeys = []
    # Run 'Begin Routine' code from FPS_counter
    tiempo_anterior = 0 
    fps = 0  # Variable para almacenar el FPS
    # keep track of which components have finished
    load_configComponents = [periphereal_region_result, key_resp_4, logs2, FPS_logs]
    for thisComponent in load_configComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "load_config" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *periphereal_region_result* updates
        
        # if periphereal_region_result is starting this frame...
        if periphereal_region_result.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            periphereal_region_result.frameNStart = frameN  # exact frame index
            periphereal_region_result.tStart = t  # local t and not account for scr refresh
            periphereal_region_result.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(periphereal_region_result, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'periphereal_region_result.started')
            # update status
            periphereal_region_result.status = STARTED
            periphereal_region_result.setAutoDraw(True)
        
        # if periphereal_region_result is active this frame...
        if periphereal_region_result.status == STARTED:
            # update params
            pass
        
        # *key_resp_4* updates
        
        # if key_resp_4 is starting this frame...
        if key_resp_4.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_4.frameNStart = frameN  # exact frame index
            key_resp_4.tStart = t  # local t and not account for scr refresh
            key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_4.status = STARTED
            # keyboard checking is just starting
            key_resp_4.clock.reset()  # now t=0
        if key_resp_4.status == STARTED:
            theseKeys = key_resp_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_4_allKeys.extend(theseKeys)
            if len(_key_resp_4_allKeys):
                key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *logs2* updates
        
        # if logs2 is starting this frame...
        if logs2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            logs2.frameNStart = frameN  # exact frame index
            logs2.tStart = t  # local t and not account for scr refresh
            logs2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(logs2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'logs2.started')
            # update status
            logs2.status = STARTED
            logs2.setAutoDraw(True)
        
        # if logs2 is active this frame...
        if logs2.status == STARTED:
            # update params
            pass
        # Run 'Each Frame' code from time_daemon
        if t>5:
            continueRoutine = False
        # Run 'Each Frame' code from FPS_counter
        tiempo_actual = t
        delta_tiempo = tiempo_actual - tiempo_anterior # tiempo desde el frame anterior
        
        if delta_tiempo > 0:
            fps = 1.0 / delta_tiempo  # Frecuencia: (1 / tiempo entre frames) (Hz)
        
        tiempo_anterior = tiempo_actual
        
        FPS_logs.text = f"FPS: {fps:.2f}"  # Mostrar con 2 decimales
        
        
        # *FPS_logs* updates
        
        # if FPS_logs is starting this frame...
        if FPS_logs.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            FPS_logs.frameNStart = frameN  # exact frame index
            FPS_logs.tStart = t  # local t and not account for scr refresh
            FPS_logs.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(FPS_logs, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'FPS_logs.started')
            # update status
            FPS_logs.status = STARTED
            FPS_logs.setAutoDraw(True)
        
        # if FPS_logs is active this frame...
        if FPS_logs.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in load_configComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "load_config" ---
    for thisComponent in load_configComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('load_config.stopped', globalClock.getTime())
    # check responses
    if key_resp_4.keys in ['', [], None]:  # No response was made
        key_resp_4.keys = None
    thisExp.addData('key_resp_4.keys',key_resp_4.keys)
    if key_resp_4.keys != None:  # we had a response
        thisExp.addData('key_resp_4.rt', key_resp_4.rt)
        thisExp.addData('key_resp_4.duration', key_resp_4.duration)
    thisExp.nextEntry()
    # the Routine "load_config" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    spatial_freq_instructions = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('instructions/spatial_frequency_staircase_instructions.xlsx'),
        seed=None, name='spatial_freq_instructions')
    thisExp.addLoop(spatial_freq_instructions)  # add the loop to the experiment
    thisSpatial_freq_instruction = spatial_freq_instructions.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisSpatial_freq_instruction.rgb)
    if thisSpatial_freq_instruction != None:
        for paramName in thisSpatial_freq_instruction:
            globals()[paramName] = thisSpatial_freq_instruction[paramName]
    
    for thisSpatial_freq_instruction in spatial_freq_instructions:
        currentLoop = spatial_freq_instructions
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisSpatial_freq_instruction.rgb)
        if thisSpatial_freq_instruction != None:
            for paramName in thisSpatial_freq_instruction:
                globals()[paramName] = thisSpatial_freq_instruction[paramName]
    # completed 1.0 repeats of 'spatial_freq_instructions'
    
    
    # set up handler to look after randomisation of conditions etc
    FFT_instructions = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('instructions/fft_staircase_instructions.xlsx'),
        seed=None, name='FFT_instructions')
    thisExp.addLoop(FFT_instructions)  # add the loop to the experiment
    thisFFT_instruction = FFT_instructions.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisFFT_instruction.rgb)
    if thisFFT_instruction != None:
        for paramName in thisFFT_instruction:
            globals()[paramName] = thisFFT_instruction[paramName]
    
    for thisFFT_instruction in FFT_instructions:
        currentLoop = FFT_instructions
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisFFT_instruction.rgb)
        if thisFFT_instruction != None:
            for paramName in thisFFT_instruction:
                globals()[paramName] = thisFFT_instruction[paramName]
    # completed 1.0 repeats of 'FFT_instructions'
    
    
    # set up handler to look after randomisation of conditions etc
    contrast_instructions = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('instructions/contrast_staircase_instructions.xlsx'),
        seed=None, name='contrast_instructions')
    thisExp.addLoop(contrast_instructions)  # add the loop to the experiment
    thisContrast_instruction = contrast_instructions.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisContrast_instruction.rgb)
    if thisContrast_instruction != None:
        for paramName in thisContrast_instruction:
            globals()[paramName] = thisContrast_instruction[paramName]
    
    for thisContrast_instruction in contrast_instructions:
        currentLoop = contrast_instructions
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisContrast_instruction.rgb)
        if thisContrast_instruction != None:
            for paramName in thisContrast_instruction:
                globals()[paramName] = thisContrast_instruction[paramName]
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'contrast_instructions'
    
    
    # set up handler to look after randomisation of conditions etc
    color_instructions = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('instructions/color_staircase_instructions.xlsx'),
        seed=None, name='color_instructions')
    thisExp.addLoop(color_instructions)  # add the loop to the experiment
    thisColor_instruction = color_instructions.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisColor_instruction.rgb)
    if thisColor_instruction != None:
        for paramName in thisColor_instruction:
            globals()[paramName] = thisColor_instruction[paramName]
    
    for thisColor_instruction in color_instructions:
        currentLoop = color_instructions
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisColor_instruction.rgb)
        if thisColor_instruction != None:
            for paramName in thisColor_instruction:
                globals()[paramName] = thisColor_instruction[paramName]
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'color_instructions'
    
    
    # set up handler to look after randomisation of conditions etc
    colors_to_test = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('colors_to_test.xlsx'),
        seed=None, name='colors_to_test')
    thisExp.addLoop(colors_to_test)  # add the loop to the experiment
    thisColors_to_test = colors_to_test.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisColors_to_test.rgb)
    if thisColors_to_test != None:
        for paramName in thisColors_to_test:
            globals()[paramName] = thisColors_to_test[paramName]
    
    for thisColors_to_test in colors_to_test:
        currentLoop = colors_to_test
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisColors_to_test.rgb)
        if thisColors_to_test != None:
            for paramName in thisColors_to_test:
                globals()[paramName] = thisColors_to_test[paramName]
    # completed 1.0 repeats of 'colors_to_test'
    
    
    # --- Prepare to start Routine "LOAD_THRESHOLDS" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('LOAD_THRESHOLDS.started', globalClock.getTime())
    # Run 'Begin Routine' code from code_22
    threshold_dict = load_thresholds_from_json()
    # keep track of which components have finished
    LOAD_THRESHOLDSComponents = []
    for thisComponent in LOAD_THRESHOLDSComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "LOAD_THRESHOLDS" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in LOAD_THRESHOLDSComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "LOAD_THRESHOLDS" ---
    for thisComponent in LOAD_THRESHOLDSComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('LOAD_THRESHOLDS.stopped', globalClock.getTime())
    # the Routine "LOAD_THRESHOLDS" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    BL1_instructions = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('instructions/BL1_instructions.xlsx'),
        seed=None, name='BL1_instructions')
    thisExp.addLoop(BL1_instructions)  # add the loop to the experiment
    thisBL1_instruction = BL1_instructions.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBL1_instruction.rgb)
    if thisBL1_instruction != None:
        for paramName in thisBL1_instruction:
            globals()[paramName] = thisBL1_instruction[paramName]
    
    for thisBL1_instruction in BL1_instructions:
        currentLoop = BL1_instructions
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBL1_instruction.rgb)
        if thisBL1_instruction != None:
            for paramName in thisBL1_instruction:
                globals()[paramName] = thisBL1_instruction[paramName]
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'BL1_instructions'
    
    
    # set up handler to look after randomisation of conditions etc
    trials_bl_1 = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('BL1.csv', selection='0:15'),
        seed=None, name='trials_bl_1')
    thisExp.addLoop(trials_bl_1)  # add the loop to the experiment
    thisTrials_bl_1 = trials_bl_1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_bl_1.rgb)
    if thisTrials_bl_1 != None:
        for paramName in thisTrials_bl_1:
            globals()[paramName] = thisTrials_bl_1[paramName]
    
    for thisTrials_bl_1 in trials_bl_1:
        currentLoop = trials_bl_1
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_bl_1.rgb)
        if thisTrials_bl_1 != None:
            for paramName in thisTrials_bl_1:
                globals()[paramName] = thisTrials_bl_1[paramName]
        
        # --- Prepare to start Routine "BL_1_SPATIAL_FREQ" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('BL_1_SPATIAL_FREQ.started', globalClock.getTime())
        dots_black_5.refreshDots()
        dots_white_5.refreshDots()
        stim.setColor([1,1,1], colorSpace='rgb')
        stim.setContrast(1.0)
        stim.setPos((stim_x, stim_y))
        stim.setSize(grating_size)
        # Run 'Begin Routine' code from code
        import math
        import random
        
        def calculate_diameter(excentricity, distance_to_screen, screen_height  = None):
            '''
            Calculates the diameter of circunference correspondint to the excentricity angle depending on the screen height and distance to screen.
            Params:
                -excentricity: angle of the excentricity in degrees
                -distance_to_screen: distance between patient and screen in meters
                -screen_height: height of the screen in meters (default is None, in this case the function will only return the diameter in unit that psychopy understands)
            Returns: 
                -diameter_unit: unit diameter (this is the diameter that psychopy understands, it should match with the diameter in meters when used)
                -diameter_m: diameter in meters (this is the real diameter it should have in the screen)
            '''
        
            import math
        
            if screen_height == None:
                diameter_m = 2 * distance_to_screen * math.sin(math.radians(excentricity))
                return None, diameter_m
            else:
                diameter_unit = (2 * distance_to_screen * math.sin(math.radians(excentricity)))/screen_height
                diameter_m = 2 * distance_to_screen * math.sin(math.radians(excentricity))
                return diameter_unit, diameter_m
        
        def calcular_posicion_stim(angulo_grados, excentricidad, altura_pantalla):
            # primero calculo el diametro en pantalla correspondiente a la excentricidad 
            diameter_unit, _ = calculate_diameter(excentricidad, 0.65, altura_pantalla)
            radius = diameter_unit / 2
            
            #hallo el punto donde mostrar el estimulo sobre la circunferencia de la excentricidad deseada
            theta = math.radians(angulo_grados)
            stim_x = radius * math.cos(theta)
            stim_y = radius * math.sin(theta)
            
            return stim_x, stim_y
        
        ####################################################
        ########____LOAD STAIRCASE TEST RESULTS____#########
        ####################################################
        #threshold_dict = load_thresholds_from_json()
        spatial_frequency_threshold = threshold_dict['spatial_frequency_threshold']
        
        
        ####################################################
        ###############____PARAMS CONFIG____################
        ####################################################
        posicion_estimulo = stim_x, stim_y = calcular_posicion_stim(posicion_angular, excentricidad, dim_y)
        diametros_central_periferica = calculate_diameter(9, 0.65, dim_y)
        diametros_estimulo = calculate_diameter(excentricidad, 0.65, dim_y)
        
        stim.sf = spatial_frequency_threshold + spatial_frequency_threshold*offset_porcentual/100
        stim.ori = orientacion
        
        #other
        gaze_position = mouse.getPosition()
        
        logs_parametros_trial.alignText='left'
        logs_parametros_trial.anchorHoriz='left'
        
        first_frame = True
        flag_skip_all = False
        flag_answer_registered = False
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # keep track of which components have finished
        BL_1_SPATIAL_FREQComponents = [dots_black_5, dots_white_5, stim, key_resp, logs_background, logs_background_2, logs, logs_parametros_trial, logs_coordenadas_mirada, gaze]
        for thisComponent in BL_1_SPATIAL_FREQComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "BL_1_SPATIAL_FREQ" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dots_black_5* updates
            
            # if dots_black_5 is starting this frame...
            if dots_black_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dots_black_5.frameNStart = frameN  # exact frame index
                dots_black_5.tStart = t  # local t and not account for scr refresh
                dots_black_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dots_black_5, 'tStartRefresh')  # time at next scr refresh
                # update status
                dots_black_5.status = STARTED
                dots_black_5.setAutoDraw(True)
            
            # if dots_black_5 is active this frame...
            if dots_black_5.status == STARTED:
                # update params
                pass
            
            # *dots_white_5* updates
            
            # if dots_white_5 is starting this frame...
            if dots_white_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dots_white_5.frameNStart = frameN  # exact frame index
                dots_white_5.tStart = t  # local t and not account for scr refresh
                dots_white_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dots_white_5, 'tStartRefresh')  # time at next scr refresh
                # update status
                dots_white_5.status = STARTED
                dots_white_5.setAutoDraw(True)
            
            # if dots_white_5 is active this frame...
            if dots_white_5.status == STARTED:
                # update params
                pass
            
            # *stim* updates
            
            # if stim is starting this frame...
            if stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                stim.frameNStart = frameN  # exact frame index
                stim.tStart = t  # local t and not account for scr refresh
                stim.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stim, 'tStartRefresh')  # time at next scr refresh
                # update status
                stim.status = STARTED
                stim.setAutoDraw(True)
            
            # if stim is active this frame...
            if stim.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from code
            ####################################################
            ##############____ON SCREEN LOGS____################
            ####################################################
            gaze_position = mouse.getPosition()
            logs_coordenadas_mirada.setText(f'{gaze_position[0]:.2f},{gaze_position[1]:.2f}')
            
            logs_parametros_trial.setText(
                f"Prueba 1 - Frecuencia espacial\n"
                f"Intento: {intento}\n"
                f"Orientación: {orientacion:.2f}\n"
                f"Excentricidad: {excentricidad}º\n"
                f"Posicion Estimulo: ({posicion_estimulo[0]:.2f}, {posicion_estimulo[1]:.2f})\n"
                f"Tamaño Estímulo: {grating_size[0]:.2f}\n"
                f"Tipo: {tipo}\n"
                f"Umbral frecuencia espacial: {spatial_frequency_threshold:.2f}\n"
                f"Offset aplicado: {offset_porcentual}\n"
                f"SF mostrado: {spatial_frequency_threshold + spatial_frequency_threshold*offset_porcentual/100:.2f}" 
            )
            
            ####################################################
            ##########____GAZE VS REGION POSITION____###########
            ####################################################
            # Calcula la distancia del ratón al centro de foveal_region
            dist_from_center = ((gaze_position[0] - foveal_region_pos[0])**2 + (gaze_position[1] - foveal_region_pos[1])**2)**0.5
            
            # Comprueba si la distancia es menor que el radio de foveal_region
            if dist_from_center <= 0.25/2:#foveal_region.radius:
                logs.setText("La mirada está dentro de la región")
            
            else:
                logs.setText("La mirada está fuera de la región")  
            
            ####################################################
            ##############____EVENTS & STATES____###############
            ####################################################
                
            # START/STOP: Verifica si se ha presionado la tecla
            flag_skip_all = False
            flag_answer_registered = False
            
            keys = event.getKeys()
            if 'space' in keys:
                flag_skip_all = True
            elif 'right' in keys:
                flag_answer_registered = True
            elif 'left' in keys:
                flag_answer_registered = True
            elif 'down' in keys:
                flag_answer_registered = True
            
            ####################################################
            ###############____TIME & NOISE____#################
            ####################################################
            
            if first_frame: # Ejecucion unica
                dots_white_5.setAutoDraw(False)
                dots_black_5.setAutoDraw(False)
                first_time = False
            
            if (t>stim_time) or flag_answer_registered: # time exceeded OR answer registered
                stim.setAutoDraw(False)
                show_noise(dots_white_5, dots_black_5, response_time)
                continueRoutine = False
                
            if flag_skip_all:
                trials_bl_1.finished = True
            
            # *key_resp* updates
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                key_resp.clock.reset()  # now t=0
            if key_resp.status == STARTED:
                theseKeys = key_resp.getKeys(keyList=['space', 'right', 'left', 'down'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
            
            # *logs_background* updates
            
            # if logs_background is starting this frame...
            if logs_background.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_background.frameNStart = frameN  # exact frame index
                logs_background.tStart = t  # local t and not account for scr refresh
                logs_background.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_background, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'logs_background.started')
                # update status
                logs_background.status = STARTED
                logs_background.setAutoDraw(True)
            
            # if logs_background is active this frame...
            if logs_background.status == STARTED:
                # update params
                pass
            
            # *logs_background_2* updates
            
            # if logs_background_2 is starting this frame...
            if logs_background_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_background_2.frameNStart = frameN  # exact frame index
                logs_background_2.tStart = t  # local t and not account for scr refresh
                logs_background_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_background_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'logs_background_2.started')
                # update status
                logs_background_2.status = STARTED
                logs_background_2.setAutoDraw(True)
            
            # if logs_background_2 is active this frame...
            if logs_background_2.status == STARTED:
                # update params
                pass
            
            # *logs* updates
            
            # if logs is starting this frame...
            if logs.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs.frameNStart = frameN  # exact frame index
                logs.tStart = t  # local t and not account for scr refresh
                logs.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'logs.started')
                # update status
                logs.status = STARTED
                logs.setAutoDraw(True)
            
            # if logs is active this frame...
            if logs.status == STARTED:
                # update params
                pass
            
            # *logs_parametros_trial* updates
            
            # if logs_parametros_trial is starting this frame...
            if logs_parametros_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_parametros_trial.frameNStart = frameN  # exact frame index
                logs_parametros_trial.tStart = t  # local t and not account for scr refresh
                logs_parametros_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_parametros_trial, 'tStartRefresh')  # time at next scr refresh
                # update status
                logs_parametros_trial.status = STARTED
                logs_parametros_trial.setAutoDraw(True)
            
            # if logs_parametros_trial is active this frame...
            if logs_parametros_trial.status == STARTED:
                # update params
                pass
            
            # *logs_coordenadas_mirada* updates
            
            # if logs_coordenadas_mirada is starting this frame...
            if logs_coordenadas_mirada.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_coordenadas_mirada.frameNStart = frameN  # exact frame index
                logs_coordenadas_mirada.tStart = t  # local t and not account for scr refresh
                logs_coordenadas_mirada.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_coordenadas_mirada, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'logs_coordenadas_mirada.started')
                # update status
                logs_coordenadas_mirada.status = STARTED
                logs_coordenadas_mirada.setAutoDraw(True)
            
            # if logs_coordenadas_mirada is active this frame...
            if logs_coordenadas_mirada.status == STARTED:
                # update params
                pass
            
            # *gaze* updates
            
            # if gaze is starting this frame...
            if gaze.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                gaze.frameNStart = frameN  # exact frame index
                gaze.tStart = t  # local t and not account for scr refresh
                gaze.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gaze, 'tStartRefresh')  # time at next scr refresh
                # update status
                gaze.status = STARTED
                gaze.setAutoDraw(True)
            
            # if gaze is active this frame...
            if gaze.status == STARTED:
                # update params
                gaze.setPos(gaze_position, log=False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in BL_1_SPATIAL_FREQComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "BL_1_SPATIAL_FREQ" ---
        for thisComponent in BL_1_SPATIAL_FREQComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('BL_1_SPATIAL_FREQ.stopped', globalClock.getTime())
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        trials_bl_1.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            trials_bl_1.addData('key_resp.rt', key_resp.rt)
            trials_bl_1.addData('key_resp.duration', key_resp.duration)
        # the Routine "BL_1_SPATIAL_FREQ" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_bl_1'
    
    
    # set up handler to look after randomisation of conditions etc
    BL2_instructions = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('instructions/BL2_instructions.xlsx'),
        seed=None, name='BL2_instructions')
    thisExp.addLoop(BL2_instructions)  # add the loop to the experiment
    thisBL2_instruction = BL2_instructions.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBL2_instruction.rgb)
    if thisBL2_instruction != None:
        for paramName in thisBL2_instruction:
            globals()[paramName] = thisBL2_instruction[paramName]
    
    for thisBL2_instruction in BL2_instructions:
        currentLoop = BL2_instructions
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBL2_instruction.rgb)
        if thisBL2_instruction != None:
            for paramName in thisBL2_instruction:
                globals()[paramName] = thisBL2_instruction[paramName]
    # completed 1.0 repeats of 'BL2_instructions'
    
    
    # set up handler to look after randomisation of conditions etc
    trials_bl_2 = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('BL2.csv'),
        seed=None, name='trials_bl_2')
    thisExp.addLoop(trials_bl_2)  # add the loop to the experiment
    thisTrials_bl_2 = trials_bl_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_bl_2.rgb)
    if thisTrials_bl_2 != None:
        for paramName in thisTrials_bl_2:
            globals()[paramName] = thisTrials_bl_2[paramName]
    
    for thisTrials_bl_2 in trials_bl_2:
        currentLoop = trials_bl_2
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_bl_2.rgb)
        if thisTrials_bl_2 != None:
            for paramName in thisTrials_bl_2:
                globals()[paramName] = thisTrials_bl_2[paramName]
        
        # --- Prepare to start Routine "BL_2_COLOR" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('BL_2_COLOR.started', globalClock.getTime())
        # Run 'Begin Routine' code from code_14
        import math
        import random
        
        def calculate_diameter(excentricity, distance_to_screen, screen_height  = None):
            '''
            Calculates the diameter of circunference correspondint to the excentricity angle depending on the screen height and distance to screen.
            Params:
                -excentricity: angle of the excentricity in degrees
                -distance_to_screen: distance between patient and screen in meters
                -screen_height: height of the screen in meters (default is None, in this case the function will only return the diameter in unit that psychopy understands)
            Returns: 
                -diameter_unit: unit diameter (this is the diameter that psychopy understands, it should match with the diameter in meters when used)
                -diameter_m: diameter in meters (this is the real diameter it should have in the screen)
            '''
        
            import math
        
            if screen_height == None:
                diameter_m = 2 * distance_to_screen * math.sin(math.radians(excentricity))
                return None, diameter_m
            else:
                diameter_unit = (2 * distance_to_screen * math.sin(math.radians(excentricity)))/screen_height
                diameter_m = 2 * distance_to_screen * math.sin(math.radians(excentricity))
                return diameter_unit, diameter_m
        
        def calcular_posicion_stim(angulo_grados, excentricidad, altura_pantalla):
            # primero calculo el diametro en pantalla correspondiente a la excentricidad 
            diameter_unit, _ = calculate_diameter(excentricidad, 0.65, altura_pantalla)
            radius = diameter_unit / 2
            
            #hallo el punto donde mostrar el estimulo sobre la circunferencia de la excentricidad deseada
            theta = math.radians(angulo_grados)
            stim_x = radius * math.cos(theta)
            stim_y = radius * math.sin(theta)
            
            return stim_x, stim_y
            
        ####################################################
        ########____LOAD STAIRCASE TEST RESULTS____#########
        ####################################################
        #threshold_dict = load_thresholds_from_json()
        frecuencia_espacial = threshold_dict['spatial_frequency_threshold']
        saturation_threshold = threshold_dict['color_threshold'][color_name]
        
        ####################################################
        ###############____PARAMS CONFIG____################
        ####################################################
        posicion_estimulo = stim_x, stim_y = calcular_posicion_stim(posicion_angular, excentricidad, dim_y)
        diametros_central_periferica = calculate_diameter(9, 0.65, dim_y)
        diametros_estimulo = calculate_diameter(excentricidad, 0.65, dim_y)
        
        #stim_6.sf = frecuencia_espacial
        #stim_6.orientation = orientacion
        stim_img.ori = orientacion
        
        #other
        gaze_position = mouse.getPosition()
        
        logs_parametros_trial_6.alignText='left'
        logs_parametros_trial_6.anchorHoriz='left'
        
        
        first_frame = True
        flag_skip_all = False
        flag_answer_registered = False
        # Run 'Begin Routine' code from gabor_generator_2
        frequency = frecuencia_espacial/2000 # division para equiparar con unidades del parche de psychopy
        
        size = 1600
        c1_hsv = [color_1_h,color_1_s,color_1_v]
        c2_hsv = [  color_1_h,
                    color_1_s+saturation_threshold + saturation_threshold*umbral_porcentual/100,
                    color_1_v]      #[color_2_r,color_2_g,color_2_b]
        
        #logs.text = f'freq = {frequency:.2f}\nc1 = ({c1[0]:.2f}, {c1[1]:.2f}, {c1[2]:.2f})\nc2 = ({c2[0]:.2f}, {c2[1]:.2f}, {c2[2]:.2f})'
        # Generar el parche de Gabor
        save_gabor_patch_image(frequency, 
                               size, 
                               normalizar_rgb(hsv_a_rgb(*c1_hsv)), 
                               normalizar_rgb(hsv_a_rgb(*c2_hsv)))
        
        key_resp_10.keys = []
        key_resp_10.rt = []
        _key_resp_10_allKeys = []
        stim_img.setImage('C:/Users/akoun/Desktop/Biocruces/siburmuin/src/begiBrainDemo/images/custom_stim.png')
        dots_white_6.refreshDots()
        dots_black_6.refreshDots()
        # keep track of which components have finished
        BL_2_COLORComponents = [key_resp_10, logs_background_9, logs_background_10, logs_7, logs_parametros_trial_6, logs_coordenadas_mirada_6, gaze_6, stim_img, dots_white_6, dots_black_6]
        for thisComponent in BL_2_COLORComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "BL_2_COLOR" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_14
            ####################################################
            ##############____ON SCREEN LOGS____################
            ####################################################
            gaze_position = mouse.getPosition()
            logs_coordenadas_mirada_6.setText(f'{gaze_position[0]:.2f},{gaze_position[1]:.2f}')
            
            logs_parametros_trial_6.setText(
                f"Prueba 2 - Color/saturación\n"
                f"Intento: {intento}\n"
                f"Orientación: {orientacion:.2f}\n"
                f"Excentricidad: {excentricidad}º\n"
                f"Posicion Estimulo: ({posicion_estimulo[0]:.2f}, {posicion_estimulo[1]:.2f})\n"
                f"Tamaño Estímulo: {grating_size[0]:.2f}\n"
                #f"Tipo: {tipo}\n"
                f"Offset (%): {umbral_porcentual:.2f}\n"
                f"Umbral color {color_name}(%): {saturation_threshold:.2f}\n"
                f"Sat. C1 (%): {color_1_s:.2f}\n"
                f"Sat. C2 (%): {color_1_s+saturation_threshold + saturation_threshold*umbral_porcentual/100:.2f}\n"
            )
            
            
            ####################################################
            ##########____GAZE VS REGION POSITION____###########
            ####################################################
            # Calcula la distancia del ratón al centro de foveal_region
            dist_from_center = ((gaze_position[0] - foveal_region_pos[0])**2 + (gaze_position[1] - foveal_region_pos[1])**2)**0.5
            
            # Comprueba si la distancia es menor que el radio de foveal_region
            if dist_from_center <= 0.25/2:#foveal_region.radius:
                logs_7.setText("La mirada está dentro de la circunferencia")
            
            else:
                logs_7.setText("La mirada está fuera de la circunferencia")
            
            
            ####################################################
            ##############____EVENTS & STATES____###############
            ####################################################
                
            # START/STOP: Verifica si se ha presionado la tecla
            flag_skip_all = False
            flag_answer_registered = False
            
            keys = event.getKeys()
            if 'space' in keys:
                flag_skip_all = True
            elif 'right' in keys:
                flag_answer_registered = True
            elif 'left' in keys:
                flag_answer_registered = True
            elif 'down' in keys:
                flag_answer_registered = True
            
            ####################################################
            ###############____TIME & NOISE____#################
            ####################################################
            
            if first_frame: # Ejecucion unica
                dots_white_6.setAutoDraw(False)
                dots_black_6.setAutoDraw(False)
                first_time = False
            
            if (t>stim_time) or flag_answer_registered: # time exceeded OR answer registered
                stim_img.setAutoDraw(False)
                show_noise(dots_white_6, dots_black_6, response_time)
                continueRoutine = False
            
            if flag_skip_all:
                trials_bl_2.finished = True
            
            # *key_resp_10* updates
            
            # if key_resp_10 is starting this frame...
            if key_resp_10.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_10.frameNStart = frameN  # exact frame index
                key_resp_10.tStart = t  # local t and not account for scr refresh
                key_resp_10.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_10, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_10.status = STARTED
                # keyboard checking is just starting
                key_resp_10.clock.reset()  # now t=0
            if key_resp_10.status == STARTED:
                theseKeys = key_resp_10.getKeys(keyList=['space', 'right', 'left'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_10_allKeys.extend(theseKeys)
                if len(_key_resp_10_allKeys):
                    key_resp_10.keys = _key_resp_10_allKeys[-1].name  # just the last key pressed
                    key_resp_10.rt = _key_resp_10_allKeys[-1].rt
                    key_resp_10.duration = _key_resp_10_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *logs_background_9* updates
            
            # if logs_background_9 is starting this frame...
            if logs_background_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_background_9.frameNStart = frameN  # exact frame index
                logs_background_9.tStart = t  # local t and not account for scr refresh
                logs_background_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_background_9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'logs_background_9.started')
                # update status
                logs_background_9.status = STARTED
                logs_background_9.setAutoDraw(True)
            
            # if logs_background_9 is active this frame...
            if logs_background_9.status == STARTED:
                # update params
                pass
            
            # *logs_background_10* updates
            
            # if logs_background_10 is starting this frame...
            if logs_background_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_background_10.frameNStart = frameN  # exact frame index
                logs_background_10.tStart = t  # local t and not account for scr refresh
                logs_background_10.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_background_10, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'logs_background_10.started')
                # update status
                logs_background_10.status = STARTED
                logs_background_10.setAutoDraw(True)
            
            # if logs_background_10 is active this frame...
            if logs_background_10.status == STARTED:
                # update params
                pass
            
            # *logs_7* updates
            
            # if logs_7 is starting this frame...
            if logs_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_7.frameNStart = frameN  # exact frame index
                logs_7.tStart = t  # local t and not account for scr refresh
                logs_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'logs_7.started')
                # update status
                logs_7.status = STARTED
                logs_7.setAutoDraw(True)
            
            # if logs_7 is active this frame...
            if logs_7.status == STARTED:
                # update params
                pass
            
            # *logs_parametros_trial_6* updates
            
            # if logs_parametros_trial_6 is starting this frame...
            if logs_parametros_trial_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_parametros_trial_6.frameNStart = frameN  # exact frame index
                logs_parametros_trial_6.tStart = t  # local t and not account for scr refresh
                logs_parametros_trial_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_parametros_trial_6, 'tStartRefresh')  # time at next scr refresh
                # update status
                logs_parametros_trial_6.status = STARTED
                logs_parametros_trial_6.setAutoDraw(True)
            
            # if logs_parametros_trial_6 is active this frame...
            if logs_parametros_trial_6.status == STARTED:
                # update params
                pass
            
            # *logs_coordenadas_mirada_6* updates
            
            # if logs_coordenadas_mirada_6 is starting this frame...
            if logs_coordenadas_mirada_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_coordenadas_mirada_6.frameNStart = frameN  # exact frame index
                logs_coordenadas_mirada_6.tStart = t  # local t and not account for scr refresh
                logs_coordenadas_mirada_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_coordenadas_mirada_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'logs_coordenadas_mirada_6.started')
                # update status
                logs_coordenadas_mirada_6.status = STARTED
                logs_coordenadas_mirada_6.setAutoDraw(True)
            
            # if logs_coordenadas_mirada_6 is active this frame...
            if logs_coordenadas_mirada_6.status == STARTED:
                # update params
                pass
            
            # *gaze_6* updates
            
            # if gaze_6 is starting this frame...
            if gaze_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                gaze_6.frameNStart = frameN  # exact frame index
                gaze_6.tStart = t  # local t and not account for scr refresh
                gaze_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gaze_6, 'tStartRefresh')  # time at next scr refresh
                # update status
                gaze_6.status = STARTED
                gaze_6.setAutoDraw(True)
            
            # if gaze_6 is active this frame...
            if gaze_6.status == STARTED:
                # update params
                gaze_6.setPos(gaze_position, log=False)
            
            # *stim_img* updates
            
            # if stim_img is starting this frame...
            if stim_img.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                stim_img.frameNStart = frameN  # exact frame index
                stim_img.tStart = t  # local t and not account for scr refresh
                stim_img.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stim_img, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'stim_img.started')
                # update status
                stim_img.status = STARTED
                stim_img.setAutoDraw(True)
            
            # if stim_img is active this frame...
            if stim_img.status == STARTED:
                # update params
                pass
            
            # *dots_white_6* updates
            
            # if dots_white_6 is starting this frame...
            if dots_white_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dots_white_6.frameNStart = frameN  # exact frame index
                dots_white_6.tStart = t  # local t and not account for scr refresh
                dots_white_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dots_white_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dots_white_6.started')
                # update status
                dots_white_6.status = STARTED
                dots_white_6.setAutoDraw(True)
            
            # if dots_white_6 is active this frame...
            if dots_white_6.status == STARTED:
                # update params
                pass
            
            # *dots_black_6* updates
            
            # if dots_black_6 is starting this frame...
            if dots_black_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dots_black_6.frameNStart = frameN  # exact frame index
                dots_black_6.tStart = t  # local t and not account for scr refresh
                dots_black_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dots_black_6, 'tStartRefresh')  # time at next scr refresh
                # update status
                dots_black_6.status = STARTED
                dots_black_6.setAutoDraw(True)
            
            # if dots_black_6 is active this frame...
            if dots_black_6.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in BL_2_COLORComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "BL_2_COLOR" ---
        for thisComponent in BL_2_COLORComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('BL_2_COLOR.stopped', globalClock.getTime())
        # check responses
        if key_resp_10.keys in ['', [], None]:  # No response was made
            key_resp_10.keys = None
        trials_bl_2.addData('key_resp_10.keys',key_resp_10.keys)
        if key_resp_10.keys != None:  # we had a response
            trials_bl_2.addData('key_resp_10.rt', key_resp_10.rt)
            trials_bl_2.addData('key_resp_10.duration', key_resp_10.duration)
        # the Routine "BL_2_COLOR" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_bl_2'
    
    
    # set up handler to look after randomisation of conditions etc
    BL3_instructions = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('instructions/BL3_instructions.xlsx'),
        seed=None, name='BL3_instructions')
    thisExp.addLoop(BL3_instructions)  # add the loop to the experiment
    thisBL3_instruction = BL3_instructions.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBL3_instruction.rgb)
    if thisBL3_instruction != None:
        for paramName in thisBL3_instruction:
            globals()[paramName] = thisBL3_instruction[paramName]
    
    for thisBL3_instruction in BL3_instructions:
        currentLoop = BL3_instructions
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBL3_instruction.rgb)
        if thisBL3_instruction != None:
            for paramName in thisBL3_instruction:
                globals()[paramName] = thisBL3_instruction[paramName]
    # completed 1.0 repeats of 'BL3_instructions'
    
    
    # set up handler to look after randomisation of conditions etc
    trials_bl_3 = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('BL3.csv', selection='0:15'),
        seed=None, name='trials_bl_3')
    thisExp.addLoop(trials_bl_3)  # add the loop to the experiment
    thisTrials_bl_3 = trials_bl_3.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_bl_3.rgb)
    if thisTrials_bl_3 != None:
        for paramName in thisTrials_bl_3:
            globals()[paramName] = thisTrials_bl_3[paramName]
    
    for thisTrials_bl_3 in trials_bl_3:
        currentLoop = trials_bl_3
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_bl_3.rgb)
        if thisTrials_bl_3 != None:
            for paramName in thisTrials_bl_3:
                globals()[paramName] = thisTrials_bl_3[paramName]
        
        # --- Prepare to start Routine "BL_3_CONTRAST" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('BL_3_CONTRAST.started', globalClock.getTime())
        stim_5.setColor([1,1,1], colorSpace='rgb')
        stim_5.setPos((stim_x, stim_y))
        stim_5.setSize(grating_size)
        # Run 'Begin Routine' code from code_8
        import math
        import random
        
        def calculate_diameter(excentricity, distance_to_screen, screen_height  = None):
            '''
            Calculates the diameter of circunference correspondint to the excentricity angle depending on the screen height and distance to screen.
            Params:
                -excentricity: angle of the excentricity in degrees
                -distance_to_screen: distance between patient and screen in meters
                -screen_height: height of the screen in meters (default is None, in this case the function will only return the diameter in unit that psychopy understands)
            Returns: 
                -diameter_unit: unit diameter (this is the diameter that psychopy understands, it should match with the diameter in meters when used)
                -diameter_m: diameter in meters (this is the real diameter it should have in the screen)
            '''
        
            import math
        
            if screen_height == None:
                diameter_m = 2 * distance_to_screen * math.sin(math.radians(excentricity))
                return None, diameter_m
            else:
                diameter_unit = (2 * distance_to_screen * math.sin(math.radians(excentricity)))/screen_height
                diameter_m = 2 * distance_to_screen * math.sin(math.radians(excentricity))
                return diameter_unit, diameter_m
        
        def calcular_posicion_stim(angulo_grados, excentricidad, altura_pantalla):
            # primero calculo el diametro en pantalla correspondiente a la excentricidad 
            diameter_unit, _ = calculate_diameter(excentricidad, 0.65, altura_pantalla)
            radius = diameter_unit / 2
            
            #hallo el punto donde mostrar el estimulo sobre la circunferencia de la excentricidad deseada
            theta = math.radians(angulo_grados)
            stim_x = radius * math.cos(theta)
            stim_y = radius * math.sin(theta)
            
            return stim_x, stim_y
        
        
        ####################################################
        ########____LOAD STAIRCASE TEST RESULTS____#########
        ####################################################
        #threshold_dict = load_thresholds_from_json()
        spatial_frequency_threshold = threshold_dict['spatial_frequency_threshold']
        contrast_threshold = threshold_dict['contrast_threshold']
        
        ####################################################
        ###############____PARAMS CONFIG____################
        ####################################################
        posicion_estimulo = stim_x, stim_y = calcular_posicion_stim(posicion_angular, excentricidad, dim_y)
        diametros_central_periferica = calculate_diameter(9, 0.65, dim_y)
        diametros_estimulo = calculate_diameter(excentricidad, 0.65, dim_y)
        
        stim_5.sf = spatial_frequency_threshold
        stim_5.contrast = contrast_threshold + contrast_threshold*offset_porcentual
        stim_5.ori = orientacion
        
        #other
        gaze_position = mouse.getPosition()
        
        logs_parametros_trial_5.alignText='left'
        logs_parametros_trial_5.anchorHoriz='left'
        
        first_frame = True
        flag_skip_all = False
        flag_answer_registered = False
        key_resp_9.keys = []
        key_resp_9.rt = []
        _key_resp_9_allKeys = []
        dots_white_7.refreshDots()
        dots_black_7.refreshDots()
        # keep track of which components have finished
        BL_3_CONTRASTComponents = [stim_5, key_resp_9, logs_background_7, logs_background_8, logs_6, logs_parametros_trial_5, logs_coordenadas_mirada_5, gaze_5, dots_white_7, dots_black_7]
        for thisComponent in BL_3_CONTRASTComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "BL_3_CONTRAST" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *stim_5* updates
            
            # if stim_5 is starting this frame...
            if stim_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                stim_5.frameNStart = frameN  # exact frame index
                stim_5.tStart = t  # local t and not account for scr refresh
                stim_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stim_5, 'tStartRefresh')  # time at next scr refresh
                # update status
                stim_5.status = STARTED
                stim_5.setAutoDraw(True)
            
            # if stim_5 is active this frame...
            if stim_5.status == STARTED:
                # update params
                pass
            
            # if stim_5 is stopping this frame...
            if stim_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > stim_5.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    stim_5.tStop = t  # not accounting for scr refresh
                    stim_5.frameNStop = frameN  # exact frame index
                    # update status
                    stim_5.status = FINISHED
                    stim_5.setAutoDraw(False)
            # Run 'Each Frame' code from code_8
            ####################################################
            ##############____ON SCREEN LOGS____################
            ####################################################
            gaze_position = mouse.getPosition()
            logs_coordenadas_mirada_5.setText(f'{gaze_position[0]:.2f},{gaze_position[1]:.2f}')
            
            logs_parametros_trial_5.setText(
                f"Prueba 3 - Contraste\n"
                f"Intento: {intento}\n"
                f"Orientación: {orientacion:.2f}\n"
                f"Excentricidad: {excentricidad}º\n"
                f"Posicion Estimulo: ({posicion_estimulo[0]:.2f}, {posicion_estimulo[1]:.2f})\n"
                f"Tamaño Estímulo: {grating_size[0]:.2f}\n"
                f"Tipo: {tipo}\n"
                f"Umbral contraste cargado: {contrast_threshold:.2f}\n"
                f"Offset aplicado: {offset_porcentual}\n"
                f"Contraste mostrado: {contrast_threshold + contrast_threshold*offset_porcentual/100:.2f}" 
            )
            
            ####################################################
            ##########____GAZE VS REGION POSITION____###########
            ####################################################
            # Calcula la distancia del ratón al centro de foveal_region
            dist_from_center = ((gaze_position[0] - foveal_region_pos[0])**2 + (gaze_position[1] - foveal_region_pos[1])**2)**0.5
            
            # Comprueba si la distancia es menor que el radio de foveal_region
            if dist_from_center <= 0.25/2:#foveal_region.radius:
                logs_6.setText("La mirada está dentro de la región")
            
            else:
                logs_6.setText("La mirada está fuera de la región")
            
            ####################################################
            ##############____EVENTS & STATES____###############
            ####################################################
                
            # START/STOP: Verifica si se ha presionado la tecla
            flag_skip_all = False
            flag_answer_registered = False
            
            keys = event.getKeys()
            if 'space' in keys:
                flag_skip_all = True
            elif 'right' in keys:
                flag_answer_registered = True
            elif 'left' in keys:
                flag_answer_registered = True
            elif 'down' in keys:
                flag_answer_registered = True
            
            ####################################################
            ###############____TIME & NOISE____#################
            ####################################################
            
            if first_frame: # Ejecucion unica
                dots_white_7.setAutoDraw(False)
                dots_black_7.setAutoDraw(False)
                first_time = False
            
            if (t>stim_time) or flag_answer_registered: # time exceeded OR answer registered
                stim_5.setAutoDraw(False)
                show_noise(dots_white_7, dots_black_7, response_time)
                continueRoutine = False
                
            if flag_skip_all:
                trials_bl_3.finished = True
            
            # *key_resp_9* updates
            
            # if key_resp_9 is starting this frame...
            if key_resp_9.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_9.frameNStart = frameN  # exact frame index
                key_resp_9.tStart = t  # local t and not account for scr refresh
                key_resp_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_9, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_9.status = STARTED
                # keyboard checking is just starting
                key_resp_9.clock.reset()  # now t=0
            if key_resp_9.status == STARTED:
                theseKeys = key_resp_9.getKeys(keyList=['space', 'right', 'left'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_9_allKeys.extend(theseKeys)
                if len(_key_resp_9_allKeys):
                    key_resp_9.keys = _key_resp_9_allKeys[-1].name  # just the last key pressed
                    key_resp_9.rt = _key_resp_9_allKeys[-1].rt
                    key_resp_9.duration = _key_resp_9_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *logs_background_7* updates
            
            # if logs_background_7 is starting this frame...
            if logs_background_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_background_7.frameNStart = frameN  # exact frame index
                logs_background_7.tStart = t  # local t and not account for scr refresh
                logs_background_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_background_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'logs_background_7.started')
                # update status
                logs_background_7.status = STARTED
                logs_background_7.setAutoDraw(True)
            
            # if logs_background_7 is active this frame...
            if logs_background_7.status == STARTED:
                # update params
                pass
            
            # *logs_background_8* updates
            
            # if logs_background_8 is starting this frame...
            if logs_background_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_background_8.frameNStart = frameN  # exact frame index
                logs_background_8.tStart = t  # local t and not account for scr refresh
                logs_background_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_background_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'logs_background_8.started')
                # update status
                logs_background_8.status = STARTED
                logs_background_8.setAutoDraw(True)
            
            # if logs_background_8 is active this frame...
            if logs_background_8.status == STARTED:
                # update params
                pass
            
            # *logs_6* updates
            
            # if logs_6 is starting this frame...
            if logs_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_6.frameNStart = frameN  # exact frame index
                logs_6.tStart = t  # local t and not account for scr refresh
                logs_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'logs_6.started')
                # update status
                logs_6.status = STARTED
                logs_6.setAutoDraw(True)
            
            # if logs_6 is active this frame...
            if logs_6.status == STARTED:
                # update params
                pass
            
            # *logs_parametros_trial_5* updates
            
            # if logs_parametros_trial_5 is starting this frame...
            if logs_parametros_trial_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_parametros_trial_5.frameNStart = frameN  # exact frame index
                logs_parametros_trial_5.tStart = t  # local t and not account for scr refresh
                logs_parametros_trial_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_parametros_trial_5, 'tStartRefresh')  # time at next scr refresh
                # update status
                logs_parametros_trial_5.status = STARTED
                logs_parametros_trial_5.setAutoDraw(True)
            
            # if logs_parametros_trial_5 is active this frame...
            if logs_parametros_trial_5.status == STARTED:
                # update params
                pass
            
            # *logs_coordenadas_mirada_5* updates
            
            # if logs_coordenadas_mirada_5 is starting this frame...
            if logs_coordenadas_mirada_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_coordenadas_mirada_5.frameNStart = frameN  # exact frame index
                logs_coordenadas_mirada_5.tStart = t  # local t and not account for scr refresh
                logs_coordenadas_mirada_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_coordenadas_mirada_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'logs_coordenadas_mirada_5.started')
                # update status
                logs_coordenadas_mirada_5.status = STARTED
                logs_coordenadas_mirada_5.setAutoDraw(True)
            
            # if logs_coordenadas_mirada_5 is active this frame...
            if logs_coordenadas_mirada_5.status == STARTED:
                # update params
                pass
            
            # *gaze_5* updates
            
            # if gaze_5 is starting this frame...
            if gaze_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                gaze_5.frameNStart = frameN  # exact frame index
                gaze_5.tStart = t  # local t and not account for scr refresh
                gaze_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gaze_5, 'tStartRefresh')  # time at next scr refresh
                # update status
                gaze_5.status = STARTED
                gaze_5.setAutoDraw(True)
            
            # if gaze_5 is active this frame...
            if gaze_5.status == STARTED:
                # update params
                gaze_5.setPos(gaze_position, log=False)
            
            # *dots_white_7* updates
            
            # if dots_white_7 is starting this frame...
            if dots_white_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dots_white_7.frameNStart = frameN  # exact frame index
                dots_white_7.tStart = t  # local t and not account for scr refresh
                dots_white_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dots_white_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dots_white_7.started')
                # update status
                dots_white_7.status = STARTED
                dots_white_7.setAutoDraw(True)
            
            # if dots_white_7 is active this frame...
            if dots_white_7.status == STARTED:
                # update params
                pass
            
            # *dots_black_7* updates
            
            # if dots_black_7 is starting this frame...
            if dots_black_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dots_black_7.frameNStart = frameN  # exact frame index
                dots_black_7.tStart = t  # local t and not account for scr refresh
                dots_black_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dots_black_7, 'tStartRefresh')  # time at next scr refresh
                # update status
                dots_black_7.status = STARTED
                dots_black_7.setAutoDraw(True)
            
            # if dots_black_7 is active this frame...
            if dots_black_7.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in BL_3_CONTRASTComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "BL_3_CONTRAST" ---
        for thisComponent in BL_3_CONTRASTComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('BL_3_CONTRAST.stopped', globalClock.getTime())
        # check responses
        if key_resp_9.keys in ['', [], None]:  # No response was made
            key_resp_9.keys = None
        trials_bl_3.addData('key_resp_9.keys',key_resp_9.keys)
        if key_resp_9.keys != None:  # we had a response
            trials_bl_3.addData('key_resp_9.rt', key_resp_9.rt)
            trials_bl_3.addData('key_resp_9.duration', key_resp_9.duration)
        # the Routine "BL_3_CONTRAST" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_bl_3'
    
    
    # set up handler to look after randomisation of conditions etc
    BL4_instructions = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('instructions/BL4_instructions.xlsx'),
        seed=None, name='BL4_instructions')
    thisExp.addLoop(BL4_instructions)  # add the loop to the experiment
    thisBL4_instruction = BL4_instructions.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBL4_instruction.rgb)
    if thisBL4_instruction != None:
        for paramName in thisBL4_instruction:
            globals()[paramName] = thisBL4_instruction[paramName]
    
    for thisBL4_instruction in BL4_instructions:
        currentLoop = BL4_instructions
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBL4_instruction.rgb)
        if thisBL4_instruction != None:
            for paramName in thisBL4_instruction:
                globals()[paramName] = thisBL4_instruction[paramName]
    # completed 1.0 repeats of 'BL4_instructions'
    
    
    # set up handler to look after randomisation of conditions etc
    trials_bl_4 = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('BL4.csv'),
        seed=None, name='trials_bl_4')
    thisExp.addLoop(trials_bl_4)  # add the loop to the experiment
    thisTrials_bl_4 = trials_bl_4.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_bl_4.rgb)
    if thisTrials_bl_4 != None:
        for paramName in thisTrials_bl_4:
            globals()[paramName] = thisTrials_bl_4[paramName]
    
    for thisTrials_bl_4 in trials_bl_4:
        currentLoop = trials_bl_4
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_bl_4.rgb)
        if thisTrials_bl_4 != None:
            for paramName in thisTrials_bl_4:
                globals()[paramName] = thisTrials_bl_4[paramName]
        
        # --- Prepare to start Routine "BL_4_TEMPORAL_FREQ" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('BL_4_TEMPORAL_FREQ.started', globalClock.getTime())
        # Run 'Begin Routine' code from code_15
        import math
        import random
        
        frecuencia_parpadeo = 30  # Hz, frecuencia de parpadeo deseada (valor inicial)
        frames_por_ciclo = int((frecuencia_monitor / frecuencia_parpadeo) / 2)
        opacidad = 1
        
        
        def calculate_diameter(excentricity, distance_to_screen, screen_height  = None):
            '''
            Calculates the diameter of circunference correspondint to the excentricity angle depending on the screen height and distance to screen.
            Params:
                -excentricity: angle of the excentricity in degrees
                -distance_to_screen: distance between patient and screen in meters
                -screen_height: height of the screen in meters (default is None, in this case the function will only return the diameter in unit that psychopy understands)
            Returns: 
                -diameter_unit: unit diameter (this is the diameter that psychopy understands, it should match with the diameter in meters when used)
                -diameter_m: diameter in meters (this is the real diameter it should have in the screen)
            '''
        
            import math
        
            if screen_height == None:
                diameter_m = 2 * distance_to_screen * math.sin(math.radians(excentricity))
                return None, diameter_m
            else:
                diameter_unit = (2 * distance_to_screen * math.sin(math.radians(excentricity)))/screen_height
                diameter_m = 2 * distance_to_screen * math.sin(math.radians(excentricity))
                return diameter_unit, diameter_m
        
        def calcular_posicion_stim(angulo_grados, excentricidad, altura_pantalla):
            # primero calculo el diametro en pantalla correspondiente a la excentricidad 
            diameter_unit, _ = calculate_diameter(excentricidad, 0.65, altura_pantalla)
            radius = diameter_unit / 2
            
            #hallo el punto donde mostrar el estimulo sobre la circunferencia de la excentricidad deseada
            theta = math.radians(angulo_grados)
            stim_x = radius * math.cos(theta)
            stim_y = radius * math.sin(theta)
            
            return stim_x, stim_y
        
        
        ####################################################
        ########____LOAD STAIRCASE TEST RESULTS____#########
        ####################################################
        spatial_frequency_threshold = threshold_dict['spatial_frequency_threshold']
        flicker_threshold = threshold_dict['flicker_threshold']
        
        
        ####################################################
        ###############____PARAMS CONFIG____################
        ####################################################
        posicion_estimulo = stim_x, stim_y = calcular_posicion_stim(posicion_angular, excentricidad, dim_y)
        diametros_central_periferica = calculate_diameter(9, 0.65, dim_y)
        diametros_estimulo = calculate_diameter(excentricidad, 0.65, dim_y)
        
        stim_7.sf = spatial_frequency_threshold
        stim_7.ori = orientacion
        
        #other
        gaze_position = mouse.getPosition()
        
        logs_parametros_trial_7.alignText='left'
        logs_parametros_trial_7.anchorHoriz='left'
        
        first_frame = True
        flag_skip_all = False
        flag_answer_registered = False
        key_resp_11.keys = []
        key_resp_11.rt = []
        _key_resp_11_allKeys = []
        stim_7.setColor([1,1,1], colorSpace='rgb')
        stim_7.setContrast(1.0)
        stim_7.setPos((stim_x, stim_y))
        stim_7.setSize(grating_size)
        dots_white_8.refreshDots()
        dots_black_8.refreshDots()
        # keep track of which components have finished
        BL_4_TEMPORAL_FREQComponents = [key_resp_11, logs_background_11, logs_background_12, logs_8, logs_parametros_trial_7, logs_coordenadas_mirada_7, stim_7, gaze_7, dots_white_8, dots_black_8]
        for thisComponent in BL_4_TEMPORAL_FREQComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "BL_4_TEMPORAL_FREQ" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_15
            ####################################################
            ##############____ON SCREEN LOGS____################
            ####################################################
            gaze_position = mouse.getPosition()
            logs_coordenadas_mirada_7.setText(f'{gaze_position[0]:.2f},{gaze_position[1]:.2f}')
            
            logs_parametros_trial_7.setText(
                f"Prueba 4 - Frecuencia Temporal\n"
                f"Intento: {intento}\n"
                f"Orientación: {orientacion:.2f}\n"
                f"Excentricidad: {excentricidad}º\n"
                f"Posicion Estimulo: ({posicion_estimulo[0]:.2f}, {posicion_estimulo[1]:.2f})\n"
                f"Tamaño Estímulo: {grating_size[0]:.2f}\n"
                f"Tipo: {tipo}\n"
                f"Umbral FFT cargado: {flicker_threshold:.2f}\n"
                f"Offset aplicado: {offset_porcentual}\n"
                f"Frecuencia mostrada: {flicker_threshold + flicker_threshold*offset_porcentual/100:.2f} FPS/Hz"
            )
            
            ####################################################
            #################_______FFT______###################
            ####################################################
            frecuencia_deseada = flicker_threshold + flicker_threshold*offset_porcentual/100
            if frecuencia_monitor/2 < frecuencia_deseada:
                print("No se puede mostrar la frecuencia deseada en el monitor actual")
            else:
                frames_por_ciclo = int((frecuencia_monitor / frecuencia_deseada) / 2)
                opacidad = 1 if (frameN % (2 * frames_por_ciclo)) < frames_por_ciclo else 0
                stim_7.opacity = opacidad
            
            ####################################################
            ##########____GAZE VS REGION POSITION____###########
            ####################################################
            # Calcula la distancia del ratón al centro de foveal_region
            dist_from_center = ((gaze_position[0] - foveal_region_pos[0])**2 + (gaze_position[1] - foveal_region_pos[1])**2)**0.5
            
            # Comprueba si la distancia es menor que el radio de foveal_region
            if dist_from_center <= 0.25/2:#foveal_region.radius:
                logs_8.setText("La mirada está dentro de la región")
            
            else:
                logs_8.setText("La mirada está fuera de la región")
            ####################################################
            ##############____EVENTS & STATES____###############
            ####################################################
                
            # START/STOP: Verifica si se ha presionado la tecla
            flag_skip_all = False
            flag_answer_registered = False
            
            keys = event.getKeys()
            if 'space' in keys:
                flag_skip_all = True
            elif 'right' in keys:
                flag_answer_registered = True
            elif 'left' in keys:
                flag_answer_registered = True
            elif 'down' in keys:
                flag_answer_registered = True
            
            ####################################################
            ###############____TIME & NOISE____#################
            ####################################################
            
            if first_frame: # Ejecucion unica
                dots_white_8.setAutoDraw(False)
                dots_black_8.setAutoDraw(False)
                first_time = False
            
            if (t>stim_time) or flag_answer_registered: # time exceeded OR answer registered
                stim.setAutoDraw(False)
                show_noise(dots_white_8, dots_black_8, response_time)
                continueRoutine = False
                
            if flag_skip_all:
                trials_bl_4.finished = True
            
            # *key_resp_11* updates
            
            # if key_resp_11 is starting this frame...
            if key_resp_11.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_11.frameNStart = frameN  # exact frame index
                key_resp_11.tStart = t  # local t and not account for scr refresh
                key_resp_11.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_11, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_11.status = STARTED
                # keyboard checking is just starting
                key_resp_11.clock.reset()  # now t=0
            if key_resp_11.status == STARTED:
                theseKeys = key_resp_11.getKeys(keyList=['space', 'right', 'left'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_11_allKeys.extend(theseKeys)
                if len(_key_resp_11_allKeys):
                    key_resp_11.keys = _key_resp_11_allKeys[-1].name  # just the last key pressed
                    key_resp_11.rt = _key_resp_11_allKeys[-1].rt
                    key_resp_11.duration = _key_resp_11_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *logs_background_11* updates
            
            # if logs_background_11 is starting this frame...
            if logs_background_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_background_11.frameNStart = frameN  # exact frame index
                logs_background_11.tStart = t  # local t and not account for scr refresh
                logs_background_11.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_background_11, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'logs_background_11.started')
                # update status
                logs_background_11.status = STARTED
                logs_background_11.setAutoDraw(True)
            
            # if logs_background_11 is active this frame...
            if logs_background_11.status == STARTED:
                # update params
                pass
            
            # *logs_background_12* updates
            
            # if logs_background_12 is starting this frame...
            if logs_background_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_background_12.frameNStart = frameN  # exact frame index
                logs_background_12.tStart = t  # local t and not account for scr refresh
                logs_background_12.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_background_12, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'logs_background_12.started')
                # update status
                logs_background_12.status = STARTED
                logs_background_12.setAutoDraw(True)
            
            # if logs_background_12 is active this frame...
            if logs_background_12.status == STARTED:
                # update params
                pass
            
            # *logs_8* updates
            
            # if logs_8 is starting this frame...
            if logs_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_8.frameNStart = frameN  # exact frame index
                logs_8.tStart = t  # local t and not account for scr refresh
                logs_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'logs_8.started')
                # update status
                logs_8.status = STARTED
                logs_8.setAutoDraw(True)
            
            # if logs_8 is active this frame...
            if logs_8.status == STARTED:
                # update params
                pass
            
            # *logs_parametros_trial_7* updates
            
            # if logs_parametros_trial_7 is starting this frame...
            if logs_parametros_trial_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_parametros_trial_7.frameNStart = frameN  # exact frame index
                logs_parametros_trial_7.tStart = t  # local t and not account for scr refresh
                logs_parametros_trial_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_parametros_trial_7, 'tStartRefresh')  # time at next scr refresh
                # update status
                logs_parametros_trial_7.status = STARTED
                logs_parametros_trial_7.setAutoDraw(True)
            
            # if logs_parametros_trial_7 is active this frame...
            if logs_parametros_trial_7.status == STARTED:
                # update params
                pass
            
            # *logs_coordenadas_mirada_7* updates
            
            # if logs_coordenadas_mirada_7 is starting this frame...
            if logs_coordenadas_mirada_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_coordenadas_mirada_7.frameNStart = frameN  # exact frame index
                logs_coordenadas_mirada_7.tStart = t  # local t and not account for scr refresh
                logs_coordenadas_mirada_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_coordenadas_mirada_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'logs_coordenadas_mirada_7.started')
                # update status
                logs_coordenadas_mirada_7.status = STARTED
                logs_coordenadas_mirada_7.setAutoDraw(True)
            
            # if logs_coordenadas_mirada_7 is active this frame...
            if logs_coordenadas_mirada_7.status == STARTED:
                # update params
                pass
            
            # *stim_7* updates
            
            # if stim_7 is starting this frame...
            if stim_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                stim_7.frameNStart = frameN  # exact frame index
                stim_7.tStart = t  # local t and not account for scr refresh
                stim_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stim_7, 'tStartRefresh')  # time at next scr refresh
                # update status
                stim_7.status = STARTED
                stim_7.setAutoDraw(True)
            
            # if stim_7 is active this frame...
            if stim_7.status == STARTED:
                # update params
                pass
            
            # if stim_7 is stopping this frame...
            if stim_7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > stim_7.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    stim_7.tStop = t  # not accounting for scr refresh
                    stim_7.frameNStop = frameN  # exact frame index
                    # update status
                    stim_7.status = FINISHED
                    stim_7.setAutoDraw(False)
            
            # *gaze_7* updates
            
            # if gaze_7 is starting this frame...
            if gaze_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                gaze_7.frameNStart = frameN  # exact frame index
                gaze_7.tStart = t  # local t and not account for scr refresh
                gaze_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gaze_7, 'tStartRefresh')  # time at next scr refresh
                # update status
                gaze_7.status = STARTED
                gaze_7.setAutoDraw(True)
            
            # if gaze_7 is active this frame...
            if gaze_7.status == STARTED:
                # update params
                gaze_7.setPos(gaze_position, log=False)
            
            # *dots_white_8* updates
            
            # if dots_white_8 is starting this frame...
            if dots_white_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dots_white_8.frameNStart = frameN  # exact frame index
                dots_white_8.tStart = t  # local t and not account for scr refresh
                dots_white_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dots_white_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dots_white_8.started')
                # update status
                dots_white_8.status = STARTED
                dots_white_8.setAutoDraw(True)
            
            # if dots_white_8 is active this frame...
            if dots_white_8.status == STARTED:
                # update params
                pass
            
            # *dots_black_8* updates
            
            # if dots_black_8 is starting this frame...
            if dots_black_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dots_black_8.frameNStart = frameN  # exact frame index
                dots_black_8.tStart = t  # local t and not account for scr refresh
                dots_black_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dots_black_8, 'tStartRefresh')  # time at next scr refresh
                # update status
                dots_black_8.status = STARTED
                dots_black_8.setAutoDraw(True)
            
            # if dots_black_8 is active this frame...
            if dots_black_8.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in BL_4_TEMPORAL_FREQComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "BL_4_TEMPORAL_FREQ" ---
        for thisComponent in BL_4_TEMPORAL_FREQComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('BL_4_TEMPORAL_FREQ.stopped', globalClock.getTime())
        # check responses
        if key_resp_11.keys in ['', [], None]:  # No response was made
            key_resp_11.keys = None
        trials_bl_4.addData('key_resp_11.keys',key_resp_11.keys)
        if key_resp_11.keys != None:  # we had a response
            trials_bl_4.addData('key_resp_11.rt', key_resp_11.rt)
            trials_bl_4.addData('key_resp_11.duration', key_resp_11.duration)
        # the Routine "BL_4_TEMPORAL_FREQ" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_bl_4'
    
    
    # set up handler to look after randomisation of conditions etc
    BL5_instructions = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('instructions/BL5_instructions.xlsx'),
        seed=None, name='BL5_instructions')
    thisExp.addLoop(BL5_instructions)  # add the loop to the experiment
    thisBL5_instruction = BL5_instructions.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBL5_instruction.rgb)
    if thisBL5_instruction != None:
        for paramName in thisBL5_instruction:
            globals()[paramName] = thisBL5_instruction[paramName]
    
    for thisBL5_instruction in BL5_instructions:
        currentLoop = BL5_instructions
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBL5_instruction.rgb)
        if thisBL5_instruction != None:
            for paramName in thisBL5_instruction:
                globals()[paramName] = thisBL5_instruction[paramName]
    # completed 1.0 repeats of 'BL5_instructions'
    
    
    # set up handler to look after randomisation of conditions etc
    trials_bl_5 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('BL5.csv'),
        seed=None, name='trials_bl_5')
    thisExp.addLoop(trials_bl_5)  # add the loop to the experiment
    thisTrials_bl_5 = trials_bl_5.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_bl_5.rgb)
    if thisTrials_bl_5 != None:
        for paramName in thisTrials_bl_5:
            globals()[paramName] = thisTrials_bl_5[paramName]
    
    for thisTrials_bl_5 in trials_bl_5:
        currentLoop = trials_bl_5
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_bl_5.rgb)
        if thisTrials_bl_5 != None:
            for paramName in thisTrials_bl_5:
                globals()[paramName] = thisTrials_bl_5[paramName]
        
        # --- Prepare to start Routine "BL_5_SEMANTIC_STIM" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('BL_5_SEMANTIC_STIM.started', globalClock.getTime())
        semantic_stim.setImage(path)
        key_resp_13.keys = []
        key_resp_13.rt = []
        _key_resp_13_allKeys = []
        # Run 'Begin Routine' code from code_17
        logs_parametros_trial_8.alignText='left'
        logs_parametros_trial_8.anchorHoriz='left'
        # keep track of which components have finished
        BL_5_SEMANTIC_STIMComponents = [semantic_stim, key_resp_13, logs_background_13, logs_parametros_trial_8, right_arrow, left_arrow, right_arrow_text, left_arrow_text]
        for thisComponent in BL_5_SEMANTIC_STIMComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "BL_5_SEMANTIC_STIM" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *semantic_stim* updates
            
            # if semantic_stim is starting this frame...
            if semantic_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                semantic_stim.frameNStart = frameN  # exact frame index
                semantic_stim.tStart = t  # local t and not account for scr refresh
                semantic_stim.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(semantic_stim, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'semantic_stim.started')
                # update status
                semantic_stim.status = STARTED
                semantic_stim.setAutoDraw(True)
            
            # if semantic_stim is active this frame...
            if semantic_stim.status == STARTED:
                # update params
                pass
            
            # *key_resp_13* updates
            waitOnFlip = False
            
            # if key_resp_13 is starting this frame...
            if key_resp_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_13.frameNStart = frameN  # exact frame index
                key_resp_13.tStart = t  # local t and not account for scr refresh
                key_resp_13.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_13, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_13.started')
                # update status
                key_resp_13.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_13.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_13.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_13.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_13.getKeys(keyList=['space', 'right', 'left'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_13_allKeys.extend(theseKeys)
                if len(_key_resp_13_allKeys):
                    key_resp_13.keys = _key_resp_13_allKeys[-1].name  # just the last key pressed
                    key_resp_13.rt = _key_resp_13_allKeys[-1].rt
                    key_resp_13.duration = _key_resp_13_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *logs_background_13* updates
            
            # if logs_background_13 is starting this frame...
            if logs_background_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_background_13.frameNStart = frameN  # exact frame index
                logs_background_13.tStart = t  # local t and not account for scr refresh
                logs_background_13.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_background_13, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'logs_background_13.started')
                # update status
                logs_background_13.status = STARTED
                logs_background_13.setAutoDraw(True)
            
            # if logs_background_13 is active this frame...
            if logs_background_13.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from code_17
            logs_parametros_trial_8.setText(
                f"Prueba 5 - Estímulos semánticos\n"
                f"Intento: {intento}\n"
                f"Excentricidad: {excentricidad}º\n"
                f"Tamaño Estímulo: {tamanyo:.2f}\n"
                f"Tipo: {tipo}\n"  
            )
            
            # *logs_parametros_trial_8* updates
            
            # if logs_parametros_trial_8 is starting this frame...
            if logs_parametros_trial_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_parametros_trial_8.frameNStart = frameN  # exact frame index
                logs_parametros_trial_8.tStart = t  # local t and not account for scr refresh
                logs_parametros_trial_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_parametros_trial_8, 'tStartRefresh')  # time at next scr refresh
                # update status
                logs_parametros_trial_8.status = STARTED
                logs_parametros_trial_8.setAutoDraw(True)
            
            # if logs_parametros_trial_8 is active this frame...
            if logs_parametros_trial_8.status == STARTED:
                # update params
                pass
            
            # *right_arrow* updates
            
            # if right_arrow is starting this frame...
            if right_arrow.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                right_arrow.frameNStart = frameN  # exact frame index
                right_arrow.tStart = t  # local t and not account for scr refresh
                right_arrow.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right_arrow, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_arrow.started')
                # update status
                right_arrow.status = STARTED
                right_arrow.setAutoDraw(True)
            
            # if right_arrow is active this frame...
            if right_arrow.status == STARTED:
                # update params
                pass
            
            # *left_arrow* updates
            
            # if left_arrow is starting this frame...
            if left_arrow.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                left_arrow.frameNStart = frameN  # exact frame index
                left_arrow.tStart = t  # local t and not account for scr refresh
                left_arrow.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_arrow, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_arrow.started')
                # update status
                left_arrow.status = STARTED
                left_arrow.setAutoDraw(True)
            
            # if left_arrow is active this frame...
            if left_arrow.status == STARTED:
                # update params
                pass
            
            # *right_arrow_text* updates
            
            # if right_arrow_text is starting this frame...
            if right_arrow_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                right_arrow_text.frameNStart = frameN  # exact frame index
                right_arrow_text.tStart = t  # local t and not account for scr refresh
                right_arrow_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right_arrow_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_arrow_text.started')
                # update status
                right_arrow_text.status = STARTED
                right_arrow_text.setAutoDraw(True)
            
            # if right_arrow_text is active this frame...
            if right_arrow_text.status == STARTED:
                # update params
                pass
            
            # *left_arrow_text* updates
            
            # if left_arrow_text is starting this frame...
            if left_arrow_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                left_arrow_text.frameNStart = frameN  # exact frame index
                left_arrow_text.tStart = t  # local t and not account for scr refresh
                left_arrow_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_arrow_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_arrow_text.started')
                # update status
                left_arrow_text.status = STARTED
                left_arrow_text.setAutoDraw(True)
            
            # if left_arrow_text is active this frame...
            if left_arrow_text.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in BL_5_SEMANTIC_STIMComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "BL_5_SEMANTIC_STIM" ---
        for thisComponent in BL_5_SEMANTIC_STIMComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('BL_5_SEMANTIC_STIM.stopped', globalClock.getTime())
        # check responses
        if key_resp_13.keys in ['', [], None]:  # No response was made
            key_resp_13.keys = None
        trials_bl_5.addData('key_resp_13.keys',key_resp_13.keys)
        if key_resp_13.keys != None:  # we had a response
            trials_bl_5.addData('key_resp_13.rt', key_resp_13.rt)
            trials_bl_5.addData('key_resp_13.duration', key_resp_13.duration)
        # the Routine "BL_5_SEMANTIC_STIM" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 1.0 repeats of 'trials_bl_5'
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
