#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on septiembre 05, 2024, at 14:45
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

# Run 'Before Experiment' code from code_20



# Run 'Before Experiment' code from code_18
from psychopy import core

def show_noise(dots_white, dots_black):
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


# Run 'Before Experiment' code from code_19


# Run 'Before Experiment' code from gabor_generator
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
# Run 'Before Experiment' code from code
frecuencia_monitor = 60
frecuencia_parpadeo = 30  # Hz, cambia este valor por la frecuencia deseada
frames_por_ciclo = int((frecuencia_monitor / frecuencia_parpadeo) / 2)
opacidad = 1
# Run 'Before Experiment' code from code_14
frecuencia_monitor = 60
frecuencia_parpadeo = 30  # Hz, cambia este valor por la frecuencia deseada
frames_por_ciclo = int((frecuencia_monitor / frecuencia_parpadeo) / 2)
opacidad = 1
# Run 'Before Experiment' code from code_stim_backend
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
    im.save(f"C:/Users/akoun/Desktop/Biocruces/siburmuin/src/begiBrainDemo/images/custom_stim.png")

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

# Run 'Before Experiment' code from code_8
frecuencia_monitor = 60
frecuencia_parpadeo = 30  # Hz, cambia este valor por la frecuencia deseada
frames_por_ciclo = int((frecuencia_monitor / frecuencia_parpadeo) / 2)
opacidad = 1
# Run 'Before Experiment' code from code_15
frecuencia_monitor = 60
frecuencia_parpadeo = 10  # Hz, cambia este valor por la frecuencia deseada
frames_por_ciclo = int((frecuencia_monitor / frecuencia_parpadeo) / 2)
opacidad = 1
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
    
    # --- Initialize components for Routine "SPATIAL_FREQ_STAIRCASE_TEST" ---
    # Run 'Begin Experiment' code from code_20
    
    
    key_resp_16 = keyboard.Keyboard()
    logs_12 = visual.TextStim(win=win, name='logs_12',
        text='Any text\n\nincluding line breaks',
        font='Open Sans',
        pos=(0, -0.45), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    grating_7 = visual.GratingStim(
        win=win, name='grating_7',
        tex='sin', mask='gauss', anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5), sf=None, phase=0.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=1.0, blendmode='avg',
        texRes=512.0, interpolate=True, depth=-3.0)
    dots_white_3 = visual.DotStim(
        win=win, name='dots_white_3',
        nDots=10000, dotSize=2.0,
        speed=0.1, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=[1.75,1], fieldAnchor='center', fieldShape='square',
        signalDots='same', noiseDots='direction',dotLife=3.0,
        color=[1.0,1.0,1.0], colorSpace='rgb', opacity=None,
        depth=-4.0)
    dots_black_3 = visual.DotStim(
        win=win, name='dots_black_3',
        nDots=10000, dotSize=2.0,
        speed=0.1, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=[1.75,1], fieldAnchor='center', fieldShape='square',
        signalDots='same', noiseDots='direction',dotLife=3.0,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None,
        depth=-5.0)
    
    # --- Initialize components for Routine "CONTRAST_STAIRCASE_TEST" ---
    # Run 'Begin Experiment' code from code_18
    
    
    key_resp_14 = keyboard.Keyboard()
    logs_10 = visual.TextStim(win=win, name='logs_10',
        text='Any text\n\nincluding line breaks',
        font='Open Sans',
        pos=(0, -0.45), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    grating = visual.GratingStim(
        win=win, name='grating',
        tex='sin', mask='gauss', anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5), sf=15.0, phase=0.0,
        color=[1,1,1], colorSpace='rgb',
        opacity=None, contrast=1.0, blendmode='avg',
        texRes=512.0, interpolate=True, depth=-3.0)
    dots_white = visual.DotStim(
        win=win, name='dots_white',
        nDots=10000, dotSize=2.0,
        speed=0.1, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=[1.75,1], fieldAnchor='center', fieldShape='square',
        signalDots='same', noiseDots='direction',dotLife=3.0,
        color=[1.0,1.0,1.0], colorSpace='rgb', opacity=None,
        depth=-4.0)
    dots_black = visual.DotStim(
        win=win, name='dots_black',
        nDots=10000, dotSize=2.0,
        speed=0.1, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=[1.75,1], fieldAnchor='center', fieldShape='square',
        signalDots='same', noiseDots='direction',dotLife=3.0,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None,
        depth=-5.0)
    
    # --- Initialize components for Routine "COLOR_STAIRCASE_TEST" ---
    key_resp_15 = keyboard.Keyboard()
    logs_11 = visual.TextStim(win=win, name='logs_11',
        text='Any text\n\nincluding line breaks',
        font='Open Sans',
        pos=(0, -0.45), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', 
        image='default.png', mask=None, anchor='center',
        ori=20.0, pos=(0, 0), size=(0.25, 0.25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=512.0, interpolate=True, depth=-4.0)
    dots_white_2 = visual.DotStim(
        win=win, name='dots_white_2',
        nDots=10000, dotSize=2.0,
        speed=0.1, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=[1.75,1], fieldAnchor='center', fieldShape='square',
        signalDots='same', noiseDots='direction',dotLife=3.0,
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None,
        depth=-5.0)
    dots_black_2 = visual.DotStim(
        win=win, name='dots_black_2',
        nDots=10000, dotSize=2.0,
        speed=0.1, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=[1.75,1], fieldAnchor='center', fieldShape='square',
        signalDots='same', noiseDots='direction',dotLife=3.0,
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None,
        depth=-6.0)
    
    # --- Initialize components for Routine "load_screen_config" ---
    # Run 'Begin Experiment' code from code_4
    periphereal_region_diameter = 0
    
    ################################
    ## CONFIGURACION MODIFICABLE: ##
    ################################
    nombre_pantalla = 'pantalla5'
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
    
    # --- Initialize components for Routine "BL_1_SPATIAL_FREQ" ---
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
    polygon = visual.ShapeStim(
        win=win, name='polygon',
        size=(0.25, 0.25), vertices='circle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    logs_background = visual.Rect(
        win=win, name='logs_background',
        width=(1, 0.3)[0], height=(1, 0.3)[1],
        ori=0.0, pos=(-0.5, 0.5), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    logs_background_2 = visual.Rect(
        win=win, name='logs_background_2',
        width=(0.5, 1)[0], height=(0.5, 1)[1],
        ori=0.0, pos=(0.75, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    logs = visual.TextStim(win=win, name='logs',
        text=None,
        font='Open Sans',
        pos=(-0.45, 0.45), height=0.035, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    logs_parametros_trial = visual.TextStim(win=win, name='logs_parametros_trial',
        text=None,
        font='Open Sans',
        pos=(0.5, 0), height=0.025, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    logs_coordenadas_mirada = visual.TextStim(win=win, name='logs_coordenadas_mirada',
        text=None,
        font='Open Sans',
        pos=(-0.45, -0.45), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    stim = visual.GratingStim(
        win=win, name='stim',
        tex='sqr', mask='gauss', anchor='center',
        ori=1.0, pos=[0,0], size=1.0, sf=1.0, phase=0.5,
        color='white', colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=512.0, interpolate=True, depth=-8.0)
    gaze = visual.ShapeStim(
        win=win, name='gaze',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
        opacity=0.4, depth=-9.0, interpolate=True)
    
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
    polygon_7 = visual.ShapeStim(
        win=win, name='polygon_7',
        size=(0.25, 0.25), vertices='circle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
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
        ori=1.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-10.0)
    
    # --- Initialize components for Routine "BL_3_CONTRAST" ---
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
    polygon_6 = visual.ShapeStim(
        win=win, name='polygon_6',
        size=(0.25, 0.25), vertices='circle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
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
    stim_5 = visual.GratingStim(
        win=win, name='stim_5',
        tex='sqr', mask='gauss', anchor='center',
        ori=1.0, pos=[0,0], size=1.0, sf=1.0, phase=0.5,
        color='white', colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=512.0, interpolate=True, depth=-8.0)
    gaze_5 = visual.ShapeStim(
        win=win, name='gaze_5',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
        opacity=0.4, depth=-9.0, interpolate=True)
    
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
    polygon_8 = visual.ShapeStim(
        win=win, name='polygon_8',
        size=(0.25, 0.25), vertices='circle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    logs_background_11 = visual.Rect(
        win=win, name='logs_background_11',
        width=(1, 0.3)[0], height=(1, 0.3)[1],
        ori=0.0, pos=(-0.5, 0.5), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    logs_background_12 = visual.Rect(
        win=win, name='logs_background_12',
        width=(0.5, 1)[0], height=(0.5, 1)[1],
        ori=0.0, pos=(0.75, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    logs_8 = visual.TextStim(win=win, name='logs_8',
        text=None,
        font='Open Sans',
        pos=(-0.45, 0.45), height=0.035, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    logs_parametros_trial_7 = visual.TextStim(win=win, name='logs_parametros_trial_7',
        text=None,
        font='Open Sans',
        pos=(0.5, 0), height=0.025, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    logs_coordenadas_mirada_7 = visual.TextStim(win=win, name='logs_coordenadas_mirada_7',
        text=None,
        font='Open Sans',
        pos=(-0.45, -0.45), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    stim_7 = visual.GratingStim(
        win=win, name='stim_7',
        tex='sqr', mask='gauss', anchor='center',
        ori=1.0, pos=[0,0], size=1.0, sf=1.0, phase=0.5,
        color='white', colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=512.0, interpolate=True, depth=-8.0)
    gaze_7 = visual.ShapeStim(
        win=win, name='gaze_7',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
        opacity=0.4, depth=-9.0, interpolate=True)
    
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
    
    # --- Prepare to start Routine "SPATIAL_FREQ_STAIRCASE_TEST" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('SPATIAL_FREQ_STAIRCASE_TEST.started', globalClock.getTime())
    # Run 'Begin Routine' code from code_20
    import csv
    
    # Variables estaticas
    sf_starting_value = 50
    sf_step_size = 15
    stop_reversals = 5#10
    
    # Inicializacion de variables que posteriormente cambian
    sf = sf_starting_value
    step = sf_step_size
    reversals = 0
    last_direction = None
    reversal_sf = []
    correct_responses = 0
    trials = []
    
    # Para almacenar las respuestas del participante
    response = None
    
    grating.sf = sf
    
    dots_white.setAutoDraw(False)
    dots_black.setAutoDraw(False)
    key_resp_16.keys = []
    key_resp_16.rt = []
    _key_resp_16_allKeys = []
    dots_white_3.refreshDots()
    dots_black_3.refreshDots()
    # keep track of which components have finished
    SPATIAL_FREQ_STAIRCASE_TESTComponents = [key_resp_16, logs_12, grating_7, dots_white_3, dots_black_3]
    for thisComponent in SPATIAL_FREQ_STAIRCASE_TESTComponents:
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
    
    # --- Run Routine "SPATIAL_FREQ_STAIRCASE_TEST" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from code_20
        keys = event.getKeys()
        
        if 's' in keys: # El paciente ve el estimulo
            response = True
        elif 'n' in keys: # El paciente no ve el eestimulo
            response = False
        
        # Lógica del staircase
        if response is not None:
            if response:  # Respuesta correcta: el paciente ve el estimulo
                correct_responses += 1
                if correct_responses == 2:  # Después de 2 respuestas correctas consecutivas
                    correct_responses = 0
                    sf = max(0, sf + step)  # Disminuir el contraste
                    if last_direction == "up":
                        reversals += 1
                        reversal_sf.append(sf)
                    last_direction = "down"
            else: 
                sf = sf - step
                correct_responses = 0
                if last_direction == "down":
                    reversals += 1
                    reversal_sf.append(sf)
                last_direction = "up"
                
            grating_7.setAutoDraw(False)
            show_noise(dots_white_3, dots_black_3)
            grating_7.setAutoDraw(True)
            # Actualizar el contraste del estímulo
            grating_7.sf = sf
            
            # Registrar la información del ensayo
            trials.append({
                'trial': len(trials) + 1,
                'spatial_frequency': sf,
                'response': response,
                'reversals': reversals
            })
            
            # Restablecer la respuesta para el siguiente ensayo
            response = None
            
            # Regla para aumentar la granularidad del test
            if (reversals % 3 == 0) and reversals != 0:
                step = step/2
                print(f"New step = {step}")
                
            # Regla de detencion
            if reversals >= stop_reversals:
                print(trials)
                # almaceno trials en 'data' para su posterior analisis
                filename = './data/sf_staircase_test_data.csv'
                with open(filename, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=['trial', 'spatial_frequency', 'response', 'reversals'])
                    writer.writeheader()
                    writer.writerows(trials)
                    
                dots_white.setAutoDraw(False)
                dots_black.setAutoDraw(False)    
                continueRoutine = False
        
        #########################################################
        #############____________LOGS_________###################
        #########################################################
        logs_12.text = f"Step Size = {step}"
        dots_white_3.setAutoDraw(False)
        dots_black_3.setAutoDraw(False)
        
        # *key_resp_16* updates
        waitOnFlip = False
        
        # if key_resp_16 is starting this frame...
        if key_resp_16.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_16.frameNStart = frameN  # exact frame index
            key_resp_16.tStart = t  # local t and not account for scr refresh
            key_resp_16.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_16, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_16.started')
            # update status
            key_resp_16.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_16.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_16.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_16.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_16.getKeys(keyList=['s','n'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_16_allKeys.extend(theseKeys)
            if len(_key_resp_16_allKeys):
                key_resp_16.keys = _key_resp_16_allKeys[-1].name  # just the last key pressed
                key_resp_16.rt = _key_resp_16_allKeys[-1].rt
                key_resp_16.duration = _key_resp_16_allKeys[-1].duration
        
        # *logs_12* updates
        
        # if logs_12 is starting this frame...
        if logs_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            logs_12.frameNStart = frameN  # exact frame index
            logs_12.tStart = t  # local t and not account for scr refresh
            logs_12.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(logs_12, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'logs_12.started')
            # update status
            logs_12.status = STARTED
            logs_12.setAutoDraw(True)
        
        # if logs_12 is active this frame...
        if logs_12.status == STARTED:
            # update params
            pass
        
        # *grating_7* updates
        
        # if grating_7 is starting this frame...
        if grating_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            grating_7.frameNStart = frameN  # exact frame index
            grating_7.tStart = t  # local t and not account for scr refresh
            grating_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(grating_7, 'tStartRefresh')  # time at next scr refresh
            # update status
            grating_7.status = STARTED
            grating_7.setAutoDraw(True)
        
        # if grating_7 is active this frame...
        if grating_7.status == STARTED:
            # update params
            pass
        
        # *dots_white_3* updates
        
        # if dots_white_3 is starting this frame...
        if dots_white_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            dots_white_3.frameNStart = frameN  # exact frame index
            dots_white_3.tStart = t  # local t and not account for scr refresh
            dots_white_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dots_white_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dots_white_3.started')
            # update status
            dots_white_3.status = STARTED
            dots_white_3.setAutoDraw(True)
        
        # if dots_white_3 is active this frame...
        if dots_white_3.status == STARTED:
            # update params
            pass
        
        # *dots_black_3* updates
        
        # if dots_black_3 is starting this frame...
        if dots_black_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            dots_black_3.frameNStart = frameN  # exact frame index
            dots_black_3.tStart = t  # local t and not account for scr refresh
            dots_black_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dots_black_3, 'tStartRefresh')  # time at next scr refresh
            # update status
            dots_black_3.status = STARTED
            dots_black_3.setAutoDraw(True)
        
        # if dots_black_3 is active this frame...
        if dots_black_3.status == STARTED:
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
        for thisComponent in SPATIAL_FREQ_STAIRCASE_TESTComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "SPATIAL_FREQ_STAIRCASE_TEST" ---
    for thisComponent in SPATIAL_FREQ_STAIRCASE_TESTComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('SPATIAL_FREQ_STAIRCASE_TEST.stopped', globalClock.getTime())
    # check responses
    if key_resp_16.keys in ['', [], None]:  # No response was made
        key_resp_16.keys = None
    thisExp.addData('key_resp_16.keys',key_resp_16.keys)
    if key_resp_16.keys != None:  # we had a response
        thisExp.addData('key_resp_16.rt', key_resp_16.rt)
        thisExp.addData('key_resp_16.duration', key_resp_16.duration)
    thisExp.nextEntry()
    # the Routine "SPATIAL_FREQ_STAIRCASE_TEST" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "CONTRAST_STAIRCASE_TEST" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('CONTRAST_STAIRCASE_TEST.started', globalClock.getTime())
    # Run 'Begin Routine' code from code_18
    import csv
    
    # Variables estaticas
    contrast_starting_value = 0.05
    contrast_step_size = 0.01
    stop_reversals = 5#10
    
    # Inicializacion de variables que posteriormente cambian
    contrast = contrast_starting_value
    step = contrast_step_size
    reversals = 0
    last_direction = None
    reversal_contrasts = []
    correct_responses = 0
    trials = []
    
    # Para almacenar las respuestas del participante
    response = None
    
    grating.contrast = contrast
    
    dots_white.setAutoDraw(False)
    dots_black.setAutoDraw(False)
    key_resp_14.keys = []
    key_resp_14.rt = []
    _key_resp_14_allKeys = []
    dots_white.refreshDots()
    dots_black.refreshDots()
    # keep track of which components have finished
    CONTRAST_STAIRCASE_TESTComponents = [key_resp_14, logs_10, grating, dots_white, dots_black]
    for thisComponent in CONTRAST_STAIRCASE_TESTComponents:
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
    
    # --- Run Routine "CONTRAST_STAIRCASE_TEST" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from code_18
        keys = event.getKeys()
        
        if 's' in keys: # El paciente ve el estimulo
            response = True
        elif 'n' in keys: # El paciente no ve el eestimulo
            response = False
        
        # Lógica del staircase
        if response is not None:
            if response:  # Respuesta correcta: el paciente ve el estimulo
                correct_responses += 1
                if correct_responses == 2:  # Después de 2 respuestas correctas consecutivas
                    correct_responses = 0
                    contrast = max(0, contrast - step)  # Disminuir el contraste
                    if last_direction == "up":
                        reversals += 1
                        reversal_contrasts.append(contrast)
                    last_direction = "down"
            else:  # Respuesta incorrecta: el paciente no ve el estimulo
                contrast += step  # Aumentar el contraste
                correct_responses = 0
                if last_direction == "down":
                    reversals += 1
                    reversal_contrasts.append(contrast)
                last_direction = "up"
                
            grating.setAutoDraw(False)
            show_noise(dots_white, dots_black)
            grating.setAutoDraw(True)
            # Actualizar el contraste del estímulo
            grating.contrast = contrast
            
            # Registrar la información del ensayo
            trials.append({
                'trial': len(trials) + 1,
                'contrast': contrast,
                'response': response,
                'reversals': reversals
            })
            
            # Restablecer la respuesta para el siguiente ensayo
            response = None
            
            # Regla para aumentar la granularidad del test
            if (reversals % 3 == 0) and reversals != 0:
                step = step/2
                print(f"New step = {step}")
                
            # Regla de detencion
            if reversals >= stop_reversals:
                print(trials)
                # almaceno trials en 'data' para su posterior analisis
                filename = './data/contrast_staircase_test_data.csv'
                with open(filename, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=['trial', 'contrast', 'response', 'reversals'])
                    writer.writeheader()
                    writer.writerows(trials)
                    
                dots_white.setAutoDraw(False)
                dots_black.setAutoDraw(False)    
                continueRoutine = False
        
        #########################################################
        #############____________LOGS_________###################
        #########################################################
        logs_10.text = f"Step Size = {step}"
        dots_white.setAutoDraw(False)
        dots_black.setAutoDraw(False)
        
        # *key_resp_14* updates
        waitOnFlip = False
        
        # if key_resp_14 is starting this frame...
        if key_resp_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_14.frameNStart = frameN  # exact frame index
            key_resp_14.tStart = t  # local t and not account for scr refresh
            key_resp_14.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_14, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_14.started')
            # update status
            key_resp_14.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_14.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_14.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_14.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_14.getKeys(keyList=['s','n'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_14_allKeys.extend(theseKeys)
            if len(_key_resp_14_allKeys):
                key_resp_14.keys = _key_resp_14_allKeys[-1].name  # just the last key pressed
                key_resp_14.rt = _key_resp_14_allKeys[-1].rt
                key_resp_14.duration = _key_resp_14_allKeys[-1].duration
        
        # *logs_10* updates
        
        # if logs_10 is starting this frame...
        if logs_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            logs_10.frameNStart = frameN  # exact frame index
            logs_10.tStart = t  # local t and not account for scr refresh
            logs_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(logs_10, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'logs_10.started')
            # update status
            logs_10.status = STARTED
            logs_10.setAutoDraw(True)
        
        # if logs_10 is active this frame...
        if logs_10.status == STARTED:
            # update params
            pass
        
        # *grating* updates
        
        # if grating is starting this frame...
        if grating.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            grating.frameNStart = frameN  # exact frame index
            grating.tStart = t  # local t and not account for scr refresh
            grating.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(grating, 'tStartRefresh')  # time at next scr refresh
            # update status
            grating.status = STARTED
            grating.setAutoDraw(True)
        
        # if grating is active this frame...
        if grating.status == STARTED:
            # update params
            pass
        
        # *dots_white* updates
        
        # if dots_white is starting this frame...
        if dots_white.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            dots_white.frameNStart = frameN  # exact frame index
            dots_white.tStart = t  # local t and not account for scr refresh
            dots_white.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dots_white, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dots_white.started')
            # update status
            dots_white.status = STARTED
            dots_white.setAutoDraw(True)
        
        # if dots_white is active this frame...
        if dots_white.status == STARTED:
            # update params
            pass
        
        # *dots_black* updates
        
        # if dots_black is starting this frame...
        if dots_black.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            dots_black.frameNStart = frameN  # exact frame index
            dots_black.tStart = t  # local t and not account for scr refresh
            dots_black.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dots_black, 'tStartRefresh')  # time at next scr refresh
            # update status
            dots_black.status = STARTED
            dots_black.setAutoDraw(True)
        
        # if dots_black is active this frame...
        if dots_black.status == STARTED:
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
        for thisComponent in CONTRAST_STAIRCASE_TESTComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "CONTRAST_STAIRCASE_TEST" ---
    for thisComponent in CONTRAST_STAIRCASE_TESTComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('CONTRAST_STAIRCASE_TEST.stopped', globalClock.getTime())
    # check responses
    if key_resp_14.keys in ['', [], None]:  # No response was made
        key_resp_14.keys = None
    thisExp.addData('key_resp_14.keys',key_resp_14.keys)
    if key_resp_14.keys != None:  # we had a response
        thisExp.addData('key_resp_14.rt', key_resp_14.rt)
        thisExp.addData('key_resp_14.duration', key_resp_14.duration)
    thisExp.nextEntry()
    # the Routine "CONTRAST_STAIRCASE_TEST" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "COLOR_STAIRCASE_TEST" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('COLOR_STAIRCASE_TEST.started', globalClock.getTime())
    # Run 'Begin Routine' code from code_19
    import csv
    # Variables estaticas
    saturation_starting_value = 55
    saturation_step_size = 5
    stop_reversals = 10
    
    # Inicializacion de variables que posteriormente cambian
    saturation = saturation_starting_value
    step = saturation_step_size
    reversals = 0
    last_direction = None
    reversal_saturations = []
    correct_responses = 0
    trials = []
    
    # Para almacenar las respuestas del participante
    response = None
    
    dots_white_2.setAutoDraw(False)
    dots_black_2.setAutoDraw(False)
    
    # Inicializacion de variables TEMPORAL --> EXTRAER COLORES DE CSV
    frequency = 0.01
    size = 500
    c1_hsv = (360, 50, 100)
    c2_hsv = (360, saturation, 100)
    
    #logs.text = f'freq = {frequency:.2f}\nc1 = ({c1[0]:.2f}, {c1[1]:.2f}, {c1[2]:.2f})\nc2 = ({c2[0]:.2f}, {c2[1]:.2f}, {c2[2]:.2f})'
    # Generar el parche de Gabor
    save_gabor_patch_image(frequency, 
                           size, 
                           normalizar_rgb(hsv_a_rgb(*c1_hsv)), 
                           normalizar_rgb(hsv_a_rgb(*c2_hsv)))
    key_resp_15.keys = []
    key_resp_15.rt = []
    _key_resp_15_allKeys = []
    # Run 'Begin Routine' code from gabor_generator
    
    
    
    
    dots_white_2.refreshDots()
    dots_black_2.refreshDots()
    # keep track of which components have finished
    COLOR_STAIRCASE_TESTComponents = [key_resp_15, logs_11, image_2, dots_white_2, dots_black_2]
    for thisComponent in COLOR_STAIRCASE_TESTComponents:
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
    
    # --- Run Routine "COLOR_STAIRCASE_TEST" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from code_19
        keys = event.getKeys()
        
        if 's' in keys: # El paciente ve el estimulo
            response = True
        elif 'n' in keys: # El paciente no ve el eestimulo
            response = False
        
        # Lógica del staircase
        if response is not None:
            if response:  # Respuesta correcta: el paciente ve el estimulo
                correct_responses += 1
                if correct_responses == 2:  # Después de 2 respuestas correctas consecutivas
                    correct_responses = 0
                    saturation = max(0, saturation - step)
                    if last_direction == "up":
                        reversals += 1
                        reversal_saturations.append(saturation)
                    last_direction = "down"
            else:  # Respuesta incorrecta: el paciente no ve el estimulo
                saturation += step  # Aumentar el contraste
                correct_responses = 0
                if last_direction == "down":
                    reversals += 1
                    reversal_saturations.append(saturation)
                last_direction = "up"
               
            image_2.setAutoDraw(False)
            show_noise(dots_white_2, dots_black_2)
            image_2.setAutoDraw(True)
            
            # Actualizar el color del estímulo
            
            #grating.contrast = contrast
        
            c2_hsv = (360, saturation, 100)
        
        #logs.text = f'freq = {frequency:.2f}\nc1 = ({c1[0]:.2f}, {c1[1]:.2f}, {c1[2]:.2f})\nc2 = ({c2[0]:.2f}, {c2[1]:.2f}, {c2[2]:.2f})'
        # Generar el parche de Gabor
        
            save_gabor_patch_image(frequency, 
                               size, 
                               normalizar_rgb(hsv_a_rgb(*c1_hsv)), 
                               normalizar_rgb(hsv_a_rgb(*c2_hsv)))
            
            # Registrar la información del ensayo
            trials.append({
                'trial': len(trials) + 1,
                'saturation': saturation,
                'response': response,
                'reversals': reversals
            })
            
            # Restablecer la respuesta para el siguiente ensayo
            response = None
            
            # Regla para aumentar la granularidad del test
            if (reversals % 3 == 0) and reversals != 0:
                step = step/2
                print(f"New step = {step}")
                
            # Regla de detencion
            if reversals >= stop_reversals:
                print(trials)
                # almaceno trials en 'data' para su posterior analisis
                filename = './data/saturation_staircase_test_data.csv'
                with open(filename, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=['trial', 'saturation', 'response', 'reversals'])
                    writer.writeheader()
                    writer.writerows(trials)
                    
                continueRoutine = False
        
            dots_white_2.setAutoDraw(False)
            dots_black_2.setAutoDraw(False)
            
        #########################################################
        #############____________LOGS_________###################
        #########################################################
        logs_11.text = f"Step Size = {step}"
        
        dots_white_2.setAutoDraw(False)
        dots_black_2.setAutoDraw(False)
        
        # *key_resp_15* updates
        waitOnFlip = False
        
        # if key_resp_15 is starting this frame...
        if key_resp_15.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_15.frameNStart = frameN  # exact frame index
            key_resp_15.tStart = t  # local t and not account for scr refresh
            key_resp_15.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_15, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_15.started')
            # update status
            key_resp_15.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_15.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_15.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_15.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_15.getKeys(keyList=['s','n'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_15_allKeys.extend(theseKeys)
            if len(_key_resp_15_allKeys):
                key_resp_15.keys = _key_resp_15_allKeys[-1].name  # just the last key pressed
                key_resp_15.rt = _key_resp_15_allKeys[-1].rt
                key_resp_15.duration = _key_resp_15_allKeys[-1].duration
        
        # *logs_11* updates
        
        # if logs_11 is starting this frame...
        if logs_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            logs_11.frameNStart = frameN  # exact frame index
            logs_11.tStart = t  # local t and not account for scr refresh
            logs_11.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(logs_11, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'logs_11.started')
            # update status
            logs_11.status = STARTED
            logs_11.setAutoDraw(True)
        
        # if logs_11 is active this frame...
        if logs_11.status == STARTED:
            # update params
            pass
        
        # *image_2* updates
        
        # if image_2 is starting this frame...
        if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_2.frameNStart = frameN  # exact frame index
            image_2.tStart = t  # local t and not account for scr refresh
            image_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_2.started')
            # update status
            image_2.status = STARTED
            image_2.setAutoDraw(True)
        
        # if image_2 is active this frame...
        if image_2.status == STARTED:
            # update params
            image_2.setImage('./images/custom_stim.png', log=False)
        
        # *dots_white_2* updates
        
        # if dots_white_2 is starting this frame...
        if dots_white_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            dots_white_2.frameNStart = frameN  # exact frame index
            dots_white_2.tStart = t  # local t and not account for scr refresh
            dots_white_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dots_white_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'dots_white_2.started')
            # update status
            dots_white_2.status = STARTED
            dots_white_2.setAutoDraw(True)
        
        # if dots_white_2 is active this frame...
        if dots_white_2.status == STARTED:
            # update params
            pass
        
        # *dots_black_2* updates
        
        # if dots_black_2 is starting this frame...
        if dots_black_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            dots_black_2.frameNStart = frameN  # exact frame index
            dots_black_2.tStart = t  # local t and not account for scr refresh
            dots_black_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(dots_black_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            dots_black_2.status = STARTED
            dots_black_2.setAutoDraw(True)
        
        # if dots_black_2 is active this frame...
        if dots_black_2.status == STARTED:
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
        for thisComponent in COLOR_STAIRCASE_TESTComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "COLOR_STAIRCASE_TEST" ---
    for thisComponent in COLOR_STAIRCASE_TESTComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('COLOR_STAIRCASE_TEST.stopped', globalClock.getTime())
    # check responses
    if key_resp_15.keys in ['', [], None]:  # No response was made
        key_resp_15.keys = None
    thisExp.addData('key_resp_15.keys',key_resp_15.keys)
    if key_resp_15.keys != None:  # we had a response
        thisExp.addData('key_resp_15.rt', key_resp_15.rt)
        thisExp.addData('key_resp_15.duration', key_resp_15.duration)
    thisExp.nextEntry()
    # the Routine "COLOR_STAIRCASE_TEST" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "load_screen_config" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('load_screen_config.started', globalClock.getTime())
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
            
            print(f'Se ha cargado la siguiente configuracion:\n'
                  f'Pantalla {nombre_pantalla} de {tamanyo_pulgadas} pulgadas con altura {dim_y} m')
            return nombre, tamanyo_pulgadas, dim_y
        else:
            print("Configuración de pantalla no encontrada.")
            return None, None, None  # Devolver None para cada valor esperado
    
    nombre, tamanyo_pulgadas, dim_y = cargar_configuracion(nombre_pantalla)
    
    
    if nombre:  # Comprobar que nombre no es None antes de usar las variables
        # Calcular el diámetro de la frontera de la periferia
       
        diameter_unit, diameter_m = calculate_diameter(alpha, distancia_eyetracker, dim_y)
        periphereal_region_diameter = diameter_unit
    
        log = f'Se ha cargado la configuracion de la {nombre_pantalla}:\n Pantalla {nombre} de {tamanyo_pulgadas} pulgadas con altura {dim_y} m.'
        print(log)
        
    logs2.setText(
                 f'Se ha cargado la configuracion de la {nombre_pantalla}:\n Pantalla {nombre} de {tamanyo_pulgadas} pulgadas con altura {dim_y} m\n' 
                 f'Para una distancia de {distancia_eyetracker} m entre el sujeto y la pantalla, el diametro debe ser de {diameter_unit:.2f} u.\n'
                 f'El diámetro equivalente es de {diameter_m:.2f} m'
                 )
    periphereal_region_result.setSize(periphereal_region_diameter)
    key_resp_4.keys = []
    key_resp_4.rt = []
    _key_resp_4_allKeys = []
    # keep track of which components have finished
    load_screen_configComponents = [periphereal_region_result, key_resp_4, logs2]
    for thisComponent in load_screen_configComponents:
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
    
    # --- Run Routine "load_screen_config" ---
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
        waitOnFlip = False
        
        # if key_resp_4 is starting this frame...
        if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_4.frameNStart = frameN  # exact frame index
            key_resp_4.tStart = t  # local t and not account for scr refresh
            key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_4.started')
            # update status
            key_resp_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_4.status == STARTED and not waitOnFlip:
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
        for thisComponent in load_screen_configComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "load_screen_config" ---
    for thisComponent in load_screen_configComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('load_screen_config.stopped', globalClock.getTime())
    # check responses
    if key_resp_4.keys in ['', [], None]:  # No response was made
        key_resp_4.keys = None
    thisExp.addData('key_resp_4.keys',key_resp_4.keys)
    if key_resp_4.keys != None:  # we had a response
        thisExp.addData('key_resp_4.rt', key_resp_4.rt)
        thisExp.addData('key_resp_4.duration', key_resp_4.duration)
    thisExp.nextEntry()
    # the Routine "load_screen_config" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
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
        ###############____PARAMS CONFIG____################
        ####################################################
        posicion_estimulo = stim_x, stim_y = calcular_posicion_stim(posicion_angular, excentricidad, dim_y)
        diametros_central_periferica = calculate_diameter(9, 0.65, dim_y)
        diametros_estimulo = calculate_diameter(excentricidad, 0.65, dim_y)
        
        stim.sf = frecuencia_espacial
        stim.orientation = orientacion
        
        #other
        gaze_position = mouse.getPosition()
        
        logs_parametros_trial.alignText='left'
        logs_parametros_trial.anchorHoriz='left'
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        stim.setColor([1,1,1], colorSpace='rgb')
        stim.setContrast(1.0)
        stim.setPos((stim_x, stim_y))
        stim.setSize(tamanyo)
        stim.setOri(orientacion)
        # keep track of which components have finished
        BL_1_SPATIAL_FREQComponents = [key_resp, polygon, logs_background, logs_background_2, logs, logs_parametros_trial, logs_coordenadas_mirada, stim, gaze]
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
                f"Tamaño Estímulo: {tamanyo:.2f}\n"
                f"Tipo: {tipo}\n"
                f"Frecuencia Espacial: {frecuencia_espacial:.2f}\n"   
            )
            '''
            logs_screen_specs.setText(
                f"Pantalla: {nombre_pantalla}\n"
                f"Altura pantalla: {dim_y} m\n"
                f"Tamaño: {tamanyo_pulgadas}\"\n"
                f"Distancia paciente - pantalla (eyetracker): {distancia_eyetracker} m\n" 
                f"Diametro umbral periferia-centro (normalizado): {periphereal_region_diameter:.2f} u\n"
                
            )
            '''
            ####################################################
            #################____SETTINGS____###################
            ####################################################
            
            
            ####################################################
            ##########____GAZE VS REGION POSITION____###########
            ####################################################
            # Calcula la distancia del ratón al centro de foveal_region
            dist_from_center = ((gaze_position[0] - foveal_region_pos[0])**2 + (gaze_position[1] - foveal_region_pos[1])**2)**0.5
            
            # Comprueba si la distancia es menor que el radio de foveal_region
            if dist_from_center <= 0.25/2:#foveal_region.radius:
                logs.setText("La mirada está dentro de la circunferencia")
            
            else:
                logs.setText("La mirada está fuera de la circunferencia")
            
            
            ####################################################
            ##############____EVENTS & STATES____###############
            ####################################################
                
            # START/STOP: Verifica si se ha presionado la tecla
            keys = event.getKeys()
            if 'space' in keys:
                pass
                #stim_x, stim_y = calcular_posicion_stim(periphereal_region_diameter)
                
            
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
                theseKeys = key_resp.getKeys(keyList=['space', 'right', 'left'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *polygon* updates
            
            # if polygon is starting this frame...
            if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon.frameNStart = frameN  # exact frame index
                polygon.tStart = t  # local t and not account for scr refresh
                polygon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                # update status
                polygon.status = STARTED
                polygon.setAutoDraw(True)
            
            # if polygon is active this frame...
            if polygon.status == STARTED:
                # update params
                pass
            
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
            
            # if stim is stopping this frame...
            if stim.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > stim.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    stim.tStop = t  # not accounting for scr refresh
                    stim.frameNStop = frameN  # exact frame index
                    # update status
                    stim.status = FINISHED
                    stim.setAutoDraw(False)
            
            # *gaze* updates
            
            # if gaze is starting this frame...
            if gaze.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                gaze.frameNStart = frameN  # exact frame index
                gaze.tStart = t  # local t and not account for scr refresh
                gaze.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gaze, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gaze.started')
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
    trials_bl_2 = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('BL2.csv', selection='0:15'),
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
        ###############____PARAMS CONFIG____################
        ####################################################
        posicion_estimulo = stim_x, stim_y = calcular_posicion_stim(posicion_angular, excentricidad, dim_y)
        diametros_central_periferica = calculate_diameter(9, 0.65, dim_y)
        diametros_estimulo = calculate_diameter(excentricidad, 0.65, dim_y)
        
        #stim_6.sf = frecuencia_espacial
        #stim_6.orientation = orientacion
        #stim_img.orientation = orientacion
        
        #other
        gaze_position = mouse.getPosition()
        
        logs_parametros_trial_6.alignText='left'
        logs_parametros_trial_6.anchorHoriz='left'
        key_resp_10.keys = []
        key_resp_10.rt = []
        _key_resp_10_allKeys = []
        # Run 'Begin Routine' code from code_stim_backend
        import random
        
        def random_frequency():
            return random.uniform(0.01, 0.1)
        
        # Parámetros aleatorios
        frequency = random_frequency()
        size = 500
        c1 = [color_1_r,color_1_g,color_1_b]
        c2 = [color_2_r,color_2_g,color_2_b]
        
        #logs.text = f'freq = {frequency:.2f}\nc1 = ({c1[0]:.2f}, {c1[1]:.2f}, {c1[2]:.2f})\nc2 = ({c2[0]:.2f}, {c2[1]:.2f}, {c2[2]:.2f})'
        # Generar el parche de Gabor
        save_gabor_patch_image(frequency, size, c1, c2)
        
        stim_img.setOri(orientacion)
        stim_img.setImage('C:/Users/akoun/Desktop/Biocruces/siburmuin/src/begiBrainDemo/images/custom_stim.png')
        # keep track of which components have finished
        BL_2_COLORComponents = [key_resp_10, polygon_7, logs_background_9, logs_background_10, logs_7, logs_parametros_trial_6, logs_coordenadas_mirada_6, gaze_6, stim_img]
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
                f"Tamaño Estímulo: {tamanyo:.2f}\n"
                f"Tipo: {tipo}\n"
                #f"Saturacion: {saturacion:.2f}\n"   
            )
            '''
            logs_screen_specs.setText(
                f"Pantalla: {nombre_pantalla}\n"
                f"Altura pantalla: {dim_y} m\n"
                f"Tamaño: {tamanyo_pulgadas}\"\n"
                f"Distancia paciente - pantalla (eyetracker): {distancia_eyetracker} m\n" 
                f"Diametro umbral periferia-centro (normalizado): {periphereal_region_diameter:.2f} u\n"
                
            )
            '''
            ####################################################
            #################____SETTINGS____###################
            ####################################################
            
            
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
            keys = event.getKeys()
            if 'space' in keys:
                pass
                #stim_x, stim_y = calcular_posicion_stim(periphereal_region_diameter)
                
            
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
            
            # *polygon_7* updates
            
            # if polygon_7 is starting this frame...
            if polygon_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon_7.frameNStart = frameN  # exact frame index
                polygon_7.tStart = t  # local t and not account for scr refresh
                polygon_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_7, 'tStartRefresh')  # time at next scr refresh
                # update status
                polygon_7.status = STARTED
                polygon_7.setAutoDraw(True)
            
            # if polygon_7 is active this frame...
            if polygon_7.status == STARTED:
                # update params
                pass
            
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
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gaze_6.started')
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
            
            # if stim_img is stopping this frame...
            if stim_img.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > stim_img.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    stim_img.tStop = t  # not accounting for scr refresh
                    stim_img.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'stim_img.stopped')
                    # update status
                    stim_img.status = FINISHED
                    stim_img.setAutoDraw(False)
            
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
        ###############____PARAMS CONFIG____################
        ####################################################
        posicion_estimulo = stim_x, stim_y = calcular_posicion_stim(posicion_angular, excentricidad, dim_y)
        diametros_central_periferica = calculate_diameter(9, 0.65, dim_y)
        diametros_estimulo = calculate_diameter(excentricidad, 0.65, dim_y)
        
        stim_5.sf = frecuencia_espacial
        stim_5.orientation = orientacion
        
        #other
        gaze_position = mouse.getPosition()
        
        logs_parametros_trial_5.alignText='left'
        logs_parametros_trial_5.anchorHoriz='left'
        key_resp_9.keys = []
        key_resp_9.rt = []
        _key_resp_9_allKeys = []
        stim_5.setColor([1,1,1], colorSpace='rgb')
        stim_5.setContrast(contraste)
        stim_5.setPos((stim_x, stim_y))
        stim_5.setSize(tamanyo)
        stim_5.setOri(orientacion)
        # keep track of which components have finished
        BL_3_CONTRASTComponents = [key_resp_9, polygon_6, logs_background_7, logs_background_8, logs_6, logs_parametros_trial_5, logs_coordenadas_mirada_5, stim_5, gaze_5]
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
                f"Tamaño Estímulo: {tamanyo:.2f}\n"
                f"Tipo: {tipo}\n"
                f"Contraste: {contraste:.2f}\n"   
            )
            '''
            logs_screen_specs.setText(
                f"Pantalla: {nombre_pantalla}\n"
                f"Altura pantalla: {dim_y} m\n"
                f"Tamaño: {tamanyo_pulgadas}\"\n"
                f"Distancia paciente - pantalla (eyetracker): {distancia_eyetracker} m\n" 
                f"Diametro umbral periferia-centro (normalizado): {periphereal_region_diameter:.2f} u\n"
                
            )
            '''
            ####################################################
            #################____SETTINGS____###################
            ####################################################
            
            
            ####################################################
            ##########____GAZE VS REGION POSITION____###########
            ####################################################
            # Calcula la distancia del ratón al centro de foveal_region
            dist_from_center = ((gaze_position[0] - foveal_region_pos[0])**2 + (gaze_position[1] - foveal_region_pos[1])**2)**0.5
            
            # Comprueba si la distancia es menor que el radio de foveal_region
            if dist_from_center <= 0.25/2:#foveal_region.radius:
                logs_6.setText("La mirada está dentro de la circunferencia")
            
            else:
                logs_6.setText("La mirada está fuera de la circunferencia")
            
            
            ####################################################
            ##############____EVENTS & STATES____###############
            ####################################################
                
            # START/STOP: Verifica si se ha presionado la tecla
            keys = event.getKeys()
            if 'space' in keys:
                pass
                #stim_x, stim_y = calcular_posicion_stim(periphereal_region_diameter)
                
            
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
            
            # *polygon_6* updates
            
            # if polygon_6 is starting this frame...
            if polygon_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon_6.frameNStart = frameN  # exact frame index
                polygon_6.tStart = t  # local t and not account for scr refresh
                polygon_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_6, 'tStartRefresh')  # time at next scr refresh
                # update status
                polygon_6.status = STARTED
                polygon_6.setAutoDraw(True)
            
            # if polygon_6 is active this frame...
            if polygon_6.status == STARTED:
                # update params
                pass
            
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
            
            # *gaze_5* updates
            
            # if gaze_5 is starting this frame...
            if gaze_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                gaze_5.frameNStart = frameN  # exact frame index
                gaze_5.tStart = t  # local t and not account for scr refresh
                gaze_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gaze_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gaze_5.started')
                # update status
                gaze_5.status = STARTED
                gaze_5.setAutoDraw(True)
            
            # if gaze_5 is active this frame...
            if gaze_5.status == STARTED:
                # update params
                gaze_5.setPos(gaze_position, log=False)
            
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
        ###############____PARAMS CONFIG____################
        ####################################################
        posicion_estimulo = stim_x, stim_y = calcular_posicion_stim(posicion_angular, excentricidad, dim_y)
        diametros_central_periferica = calculate_diameter(9, 0.65, dim_y)
        diametros_estimulo = calculate_diameter(excentricidad, 0.65, dim_y)
        
        stim_7.sf = frecuencia_espacial
        stim_7.orientation = orientacion
        
        #other
        gaze_position = mouse.getPosition()
        
        logs_parametros_trial_7.alignText='left'
        logs_parametros_trial_7.anchorHoriz='left'
        key_resp_11.keys = []
        key_resp_11.rt = []
        _key_resp_11_allKeys = []
        stim_7.setColor([1,1,1], colorSpace='rgb')
        stim_7.setContrast(1.0)
        stim_7.setPos((stim_x, stim_y))
        stim_7.setSize(tamanyo)
        stim_7.setOri(orientacion)
        # keep track of which components have finished
        BL_4_TEMPORAL_FREQComponents = [key_resp_11, polygon_8, logs_background_11, logs_background_12, logs_8, logs_parametros_trial_7, logs_coordenadas_mirada_7, stim_7, gaze_7]
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
                f"Tamaño Estímulo: {tamanyo:.2f}\n"
                f"Tipo: {tipo}\n"
                f"FFT: {FFT:.2f}\n"   
            )
            '''
            logs_screen_specs.setText(
                f"Pantalla: {nombre_pantalla}\n"
                f"Altura pantalla: {dim_y} m\n"
                f"Tamaño: {tamanyo_pulgadas}\"\n"
                f"Distancia paciente - pantalla (eyetracker): {distancia_eyetracker} m\n" 
                f"Diametro umbral periferia-centro (normalizado): {periphereal_region_diameter:.2f} u\n"
                
            )
            '''
            ####################################################
            #################_______FFT______###################
            ####################################################
            frames_por_ciclo = int((frecuencia_monitor / FFT) / 2)
            opacidad = 1 if (frameN % (2 * frames_por_ciclo)) < frames_por_ciclo else 0
            stim_7.opacity = opacidad
            
            ####################################################
            ##########____GAZE VS REGION POSITION____###########
            ####################################################
            # Calcula la distancia del ratón al centro de foveal_region
            dist_from_center = ((gaze_position[0] - foveal_region_pos[0])**2 + (gaze_position[1] - foveal_region_pos[1])**2)**0.5
            
            # Comprueba si la distancia es menor que el radio de foveal_region
            if dist_from_center <= 0.25/2:#foveal_region.radius:
                logs_8.setText("La mirada está dentro de la circunferencia")
            
            else:
                logs_8.setText("La mirada está fuera de la circunferencia")
            
            
            ####################################################
            ##############____EVENTS & STATES____###############
            ####################################################
                
            # START/STOP: Verifica si se ha presionado la tecla
            keys = event.getKeys()
            if 'space' in keys:
                pass
                #stim_x, stim_y = calcular_posicion_stim(periphereal_region_diameter)
                
            
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
            
            # *polygon_8* updates
            
            # if polygon_8 is starting this frame...
            if polygon_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon_8.frameNStart = frameN  # exact frame index
                polygon_8.tStart = t  # local t and not account for scr refresh
                polygon_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_8, 'tStartRefresh')  # time at next scr refresh
                # update status
                polygon_8.status = STARTED
                polygon_8.setAutoDraw(True)
            
            # if polygon_8 is active this frame...
            if polygon_8.status == STARTED:
                # update params
                pass
            
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
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gaze_7.started')
                # update status
                gaze_7.status = STARTED
                gaze_7.setAutoDraw(True)
            
            # if gaze_7 is active this frame...
            if gaze_7.status == STARTED:
                # update params
                gaze_7.setPos(gaze_position, log=False)
            
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
    trials = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('BL5.csv'),
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
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
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
    # completed 1.0 repeats of 'trials'
    
    
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
