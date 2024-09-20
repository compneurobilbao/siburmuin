#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on septiembre 20, 2024, at 13:26
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

#GLOBAL VARIABLES

noise_dots = 25000
grating_size = (0.5,0.5)
n_reversals_to_average = 4
stop_reversals = 5

# FUNCTIONS

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
        opacity=None, depth=-2.0, interpolate=True)
    logs_background_2 = visual.Rect(
        win=win, name='logs_background_2',
        width=(0.5, 1)[0], height=(0.5, 1)[1],
        ori=0.0, pos=(0.75, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    logs = visual.TextStim(win=win, name='logs',
        text=None,
        font='Open Sans',
        pos=(-0.45, 0.45), height=0.035, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    logs_parametros_trial = visual.TextStim(win=win, name='logs_parametros_trial',
        text=None,
        font='Open Sans',
        pos=(0.5, 0), height=0.025, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    logs_coordenadas_mirada = visual.TextStim(win=win, name='logs_coordenadas_mirada',
        text=None,
        font='Open Sans',
        pos=(-0.45, -0.45), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    stim = visual.GratingStim(
        win=win, name='stim',
        tex='sqr', mask='circle', anchor='center',
        ori=1.0, pos=[0,0], size=1.0, sf=1.0, phase=0.5,
        color='white', colorSpace='rgb',
        opacity=1.0, contrast=1.0, blendmode='avg',
        texRes=512.0, interpolate=True, depth=-7.0)
    gaze = visual.ShapeStim(
        win=win, name='gaze',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
        opacity=0.4, depth=-8.0, interpolate=True)
    
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
        stim.setSize(grating_size)
        stim.setOri(orientacion)
        # keep track of which components have finished
        BL_1_SPATIAL_FREQComponents = [key_resp, logs_background, logs_background_2, logs, logs_parametros_trial, logs_coordenadas_mirada, stim, gaze]
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
                f"Tamaño Estímulo: {grating_size[0]:.2f}\n"
                f"Tipo: {tipo}\n"
                f"Umbral frecuencia espacial: {spatial_frequency_threshold:.2f}\n"
                f"Offset aplicado: {offset_porcentual}\n"
                f"SF mostrado: {spatial_frequency_threshold + spatial_frequency_threshold*offset_porcentual/100:.2f}" 
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
                logs.setText("La mirada está dentro de la región")
            
            else:
                logs.setText("La mirada está fuera de la región")
            
            
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
