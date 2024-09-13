#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on septiembre 13, 2024, at 14:11
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
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
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

# Run 'Before Experiment' code from code
# GLOBAL
tamanyo_estimulo = (0.2, 0.2)

tiempo_descanso = 5


# FIJACION CENTRAL
central_fixation_routine_time = 10


# SACADAS
sacade_routine_time = 10
pos1 = (0,0)
pos2 = (-0.7,0)
pos3 = (0.7,0.45)
pos4 = (-0.7,-0.45)
pos5 = (-0.7,0.45)

positions_list = (pos1,pos2,pos3,pos4,pos5,pos1)

# SEGUIMIENTO

# LECTURA
texto_lectura = "El perro corre por el parque. Los niños juegan con la pelota. Ana fue al mercado. Compró frutas frescas y un ramo de flores. Al volver a casa, preparó una ensalada para la cena. El sol brilla en el cielo. El cielo es azul. El viento mueve las hojas de los árboles.\n"
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'eye_tracking_protocol_demo'  # from the Builder filename that created this script
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
        originPath='C:\\Users\\akoun\\Desktop\\Biocruces\\siburmuin\\src\\eye_tracking_test\\eye_tracking_protocol_demo_lastrun.py',
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
            size=[1440, 900], fullscr=True, screen=1,
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
    win.mouseVisible = False
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
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
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
    
    # --- Initialize components for Routine "STATIC_VARIABLES" ---
    
    # --- Initialize components for Routine "INSTRUCTIONS" ---
    logo_bio_2 = visual.ImageStim(
        win=win,
        name='logo_bio_2', 
        image='images/BIOBIZKAIA_horizontal_CMYK.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.45, 0.35), size=(0.4, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    logo_compneurolab_2 = visual.ImageStim(
        win=win,
        name='logo_compneurolab_2', 
        image='images/compneuro_horizontal.png', mask=None, anchor='center',
        ori=0.0, pos=(0.45, 0.35), size=(0.6, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    text_title_2 = visual.TextStim(win=win, name='text_title_2',
        text='ANÁLISIS DE MOVIMIENTO OCULAR',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_instructions_2 = visual.TextStim(win=win, name='text_instructions_2',
        text=None,
        font='Open Sans',
        pos=(0, -0.20), height=0.035, wrapWidth=1.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    button_next_instruction_2 = visual.ButtonStim(win, 
        text='Siguiente -->', font='Arvo',
        pos=(0.5, -0.4),
        letterHeight=0.03,
        size=(0.25, 0.15), borderWidth=0.1,
        fillColor=[-1.0000, 0.0039, -1.0000], borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_next_instruction_2',
        depth=-5
    )
    button_next_instruction_2.buttonClock = core.Clock()
    button_previous_instruction_2 = visual.ButtonStim(win, 
        text='<--Anterior', font='Arvo',
        pos=(-0.5, -0.4),
        letterHeight=0.03,
        size=(0.25, 0.15), borderWidth=0.1,
        fillColor=[-1.0000, 0.0039, -1.0000], borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_previous_instruction_2',
        depth=-6
    )
    button_previous_instruction_2.buttonClock = core.Clock()
    key_resp_skip_instructions_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "FIJACION_CENTRAL" ---
    # Run 'Begin Experiment' code from mouse_as_gaze
    from psychopy.iohub import launchHubServer
    
    io = launchHubServer()
    mouse = io.devices.mouse
    gaze_position = mouse.getPosition()
    polygon = visual.ShapeStim(
        win=win, name='polygon',
        size=tamanyo_estimulo, vertices='circle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    gaze = visual.ShapeStim(
        win=win, name='gaze',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[1.0000, -1.0000, -1.0000], fillColor=[1.0000, -1.0000, -1.0000],
        opacity=0.4, depth=-3.0, interpolate=True)
    GP_logs_2 = visual.TextStim(win=win, name='GP_logs_2',
        text=None,
        font='Open Sans',
        pos=(-0.45, 0.35), height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    GP_logs_3 = visual.TextStim(win=win, name='GP_logs_3',
        text=None,
        font='Open Sans',
        pos=(0.45, -0.35), height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "DESCANSO" ---
    text_countdown = visual.TextStim(win=win, name='text_countdown',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    texto_descanso = visual.TextStim(win=win, name='texto_descanso',
        text='Descanso',
        font='Open Sans',
        pos=(0, 0.25), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "SACADAS" ---
    # Run 'Begin Experiment' code from positions
    actual_position_index = 0
    sacade_stimuli = visual.ShapeStim(
        win=win, name='sacade_stimuli',
        size=tamanyo_estimulo, vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[-1.0000, -1.0000, 1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    
    # --- Initialize components for Routine "DESCANSO" ---
    text_countdown = visual.TextStim(win=win, name='text_countdown',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    texto_descanso = visual.TextStim(win=win, name='texto_descanso',
        text='Descanso',
        font='Open Sans',
        pos=(0, 0.25), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "SEGUIMIENTO" ---
    moving_stimuli = visual.ShapeStim(
        win=win, name='moving_stimuli',
        size=tamanyo_estimulo, vertices='circle',
        ori=0.0, pos=None, anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor=[1.0000, -0.0039, -0.3725],
        opacity=None, depth=0.0, interpolate=True)
    # Run 'Begin Experiment' code from movement_control_backend
    # Definir una lista de puntos [(x, y)] por los que pasará el estímulo
    puntos = [(-0.89, 0), (-0.5, 0.3), (0, 0), (0.5, -0.3), (0.89, 0)]
    
    # Índice del punto actual
    indice_punto = 0
    
    # Posición inicial
    moving_stimuli.pos = puntos[indice_punto]
    
    velocidad = 0.005  # Velocidad del movimiento
    
    # --- Initialize components for Routine "DESCANSO" ---
    text_countdown = visual.TextStim(win=win, name='text_countdown',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    texto_descanso = visual.TextStim(win=win, name='texto_descanso',
        text='Descanso',
        font='Open Sans',
        pos=(0, 0.25), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "LECTURA" ---
    text_lectura = visual.TextStim(win=win, name='text_lectura',
        text=texto_lectura,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
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
    
    # --- Prepare to start Routine "STATIC_VARIABLES" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('STATIC_VARIABLES.started', globalClock.getTime())
    # keep track of which components have finished
    STATIC_VARIABLESComponents = []
    for thisComponent in STATIC_VARIABLESComponents:
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
    
    # --- Run Routine "STATIC_VARIABLES" ---
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
        for thisComponent in STATIC_VARIABLESComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "STATIC_VARIABLES" ---
    for thisComponent in STATIC_VARIABLESComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('STATIC_VARIABLES.stopped', globalClock.getTime())
    # the Routine "STATIC_VARIABLES" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    instructions = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('instructions/instructions_1.xlsx'),
        seed=None, name='instructions')
    thisExp.addLoop(instructions)  # add the loop to the experiment
    thisInstruction = instructions.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisInstruction.rgb)
    if thisInstruction != None:
        for paramName in thisInstruction:
            globals()[paramName] = thisInstruction[paramName]
    
    for thisInstruction in instructions:
        currentLoop = instructions
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
        # abbreviate parameter names if possible (e.g. rgb = thisInstruction.rgb)
        if thisInstruction != None:
            for paramName in thisInstruction:
                globals()[paramName] = thisInstruction[paramName]
        
        # --- Prepare to start Routine "INSTRUCTIONS" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('INSTRUCTIONS.started', globalClock.getTime())
        # Run 'Begin Routine' code from code_9
        instruction_no = 0
        messages_instructions = [title]
        for i in range(1, 6):
            var_name = f"instruction_{i}"
            if var_name in globals():
                instruction = globals()[var_name]
                if instruction: # Si la instrucción no esta vacía se añade a la lista que aparecera por pantalla
                    messages_instructions.append(instruction)
        print(f'Lista de instrucciones cargada: {messages_instructions}')
        # reset button_next_instruction_2 to account for continued clicks & clear times on/off
        button_next_instruction_2.reset()
        # reset button_previous_instruction_2 to account for continued clicks & clear times on/off
        button_previous_instruction_2.reset()
        key_resp_skip_instructions_2.keys = []
        key_resp_skip_instructions_2.rt = []
        _key_resp_skip_instructions_2_allKeys = []
        # keep track of which components have finished
        INSTRUCTIONSComponents = [logo_bio_2, logo_compneurolab_2, text_title_2, text_instructions_2, button_next_instruction_2, button_previous_instruction_2, key_resp_skip_instructions_2]
        for thisComponent in INSTRUCTIONSComponents:
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
        
        # --- Run Routine "INSTRUCTIONS" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *logo_bio_2* updates
            
            # if logo_bio_2 is starting this frame...
            if logo_bio_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logo_bio_2.frameNStart = frameN  # exact frame index
                logo_bio_2.tStart = t  # local t and not account for scr refresh
                logo_bio_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logo_bio_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                logo_bio_2.status = STARTED
                logo_bio_2.setAutoDraw(True)
            
            # if logo_bio_2 is active this frame...
            if logo_bio_2.status == STARTED:
                # update params
                pass
            
            # *logo_compneurolab_2* updates
            
            # if logo_compneurolab_2 is starting this frame...
            if logo_compneurolab_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logo_compneurolab_2.frameNStart = frameN  # exact frame index
                logo_compneurolab_2.tStart = t  # local t and not account for scr refresh
                logo_compneurolab_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logo_compneurolab_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                logo_compneurolab_2.status = STARTED
                logo_compneurolab_2.setAutoDraw(True)
            
            # if logo_compneurolab_2 is active this frame...
            if logo_compneurolab_2.status == STARTED:
                # update params
                pass
            
            # *text_title_2* updates
            
            # if text_title_2 is starting this frame...
            if text_title_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_title_2.frameNStart = frameN  # exact frame index
                text_title_2.tStart = t  # local t and not account for scr refresh
                text_title_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_title_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_title_2.status = STARTED
                text_title_2.setAutoDraw(True)
            
            # if text_title_2 is active this frame...
            if text_title_2.status == STARTED:
                # update params
                pass
            
            # *text_instructions_2* updates
            
            # if text_instructions_2 is starting this frame...
            if text_instructions_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_instructions_2.frameNStart = frameN  # exact frame index
                text_instructions_2.tStart = t  # local t and not account for scr refresh
                text_instructions_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_instructions_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_instructions_2.status = STARTED
                text_instructions_2.setAutoDraw(True)
            
            # if text_instructions_2 is active this frame...
            if text_instructions_2.status == STARTED:
                # update params
                text_instructions_2.setText('', log=False)
            # Run 'Each Frame' code from code_9
            text_instructions_2.text = messages_instructions[instruction_no]
                
            if instruction_no == (len(messages_instructions) - 1):
                button_next_instruction_2.opacity = 0
                #button_next_instruction.status = PAUSED
            else:
                button_next_instruction_2.opacity = 1.0
                #button_next_instruction.status = STARTED
            
            if instruction_no == 0:
                button_previous_instruction_2.opacity = 0
                #button_previous_instruction.status = PAUSED
            else:
                button_previous_instruction_2.opacity = 1.0
                #button_previous_instruction.status = STARTED
            
            ###################################################
            ####________________EVENTS_____________________####
            ###################################################
            
            keys = event.getKeys()  # Cada llamada al buffer lo vaciía (teoricamente)
            
            if 'space' in keys:
                continueRoutine = False
            # *button_next_instruction_2* updates
            
            # if button_next_instruction_2 is starting this frame...
            if button_next_instruction_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                button_next_instruction_2.frameNStart = frameN  # exact frame index
                button_next_instruction_2.tStart = t  # local t and not account for scr refresh
                button_next_instruction_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(button_next_instruction_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                button_next_instruction_2.status = STARTED
                button_next_instruction_2.setAutoDraw(True)
            
            # if button_next_instruction_2 is active this frame...
            if button_next_instruction_2.status == STARTED:
                # update params
                pass
                # check whether button_next_instruction_2 has been pressed
                if button_next_instruction_2.isClicked:
                    if not button_next_instruction_2.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        button_next_instruction_2.timesOn.append(button_next_instruction_2.buttonClock.getTime())
                        button_next_instruction_2.timesOff.append(button_next_instruction_2.buttonClock.getTime())
                    elif len(button_next_instruction_2.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        button_next_instruction_2.timesOff[-1] = button_next_instruction_2.buttonClock.getTime()
                    if not button_next_instruction_2.wasClicked:
                        # run callback code when button_next_instruction_2 is clicked
                        if instruction_no < len(messages_instructions)-1:
                            instruction_no+=1
            # take note of whether button_next_instruction_2 was clicked, so that next frame we know if clicks are new
            button_next_instruction_2.wasClicked = button_next_instruction_2.isClicked and button_next_instruction_2.status == STARTED
            # *button_previous_instruction_2* updates
            
            # if button_previous_instruction_2 is starting this frame...
            if button_previous_instruction_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                button_previous_instruction_2.frameNStart = frameN  # exact frame index
                button_previous_instruction_2.tStart = t  # local t and not account for scr refresh
                button_previous_instruction_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(button_previous_instruction_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'button_previous_instruction_2.started')
                # update status
                button_previous_instruction_2.status = STARTED
                button_previous_instruction_2.setAutoDraw(True)
            
            # if button_previous_instruction_2 is active this frame...
            if button_previous_instruction_2.status == STARTED:
                # update params
                pass
                # check whether button_previous_instruction_2 has been pressed
                if button_previous_instruction_2.isClicked:
                    if not button_previous_instruction_2.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        button_previous_instruction_2.timesOn.append(button_previous_instruction_2.buttonClock.getTime())
                        button_previous_instruction_2.timesOff.append(button_previous_instruction_2.buttonClock.getTime())
                    elif len(button_previous_instruction_2.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        button_previous_instruction_2.timesOff[-1] = button_previous_instruction_2.buttonClock.getTime()
                    if not button_previous_instruction_2.wasClicked:
                        # run callback code when button_previous_instruction_2 is clicked
                        if 0 < instruction_no:
                            instruction_no-=1
            # take note of whether button_previous_instruction_2 was clicked, so that next frame we know if clicks are new
            button_previous_instruction_2.wasClicked = button_previous_instruction_2.isClicked and button_previous_instruction_2.status == STARTED
            
            # *key_resp_skip_instructions_2* updates
            waitOnFlip = False
            
            # if key_resp_skip_instructions_2 is starting this frame...
            if key_resp_skip_instructions_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_skip_instructions_2.frameNStart = frameN  # exact frame index
                key_resp_skip_instructions_2.tStart = t  # local t and not account for scr refresh
                key_resp_skip_instructions_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_skip_instructions_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_skip_instructions_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_skip_instructions_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_skip_instructions_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_skip_instructions_2.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_skip_instructions_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_skip_instructions_2_allKeys.extend(theseKeys)
                if len(_key_resp_skip_instructions_2_allKeys):
                    key_resp_skip_instructions_2.keys = _key_resp_skip_instructions_2_allKeys[-1].name  # just the last key pressed
                    key_resp_skip_instructions_2.rt = _key_resp_skip_instructions_2_allKeys[-1].rt
                    key_resp_skip_instructions_2.duration = _key_resp_skip_instructions_2_allKeys[-1].duration
                    # a response ends the routine
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
            for thisComponent in INSTRUCTIONSComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "INSTRUCTIONS" ---
        for thisComponent in INSTRUCTIONSComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('INSTRUCTIONS.stopped', globalClock.getTime())
        instructions.addData('button_next_instruction_2.numClicks', button_next_instruction_2.numClicks)
        if button_next_instruction_2.numClicks:
           instructions.addData('button_next_instruction_2.timesOn', button_next_instruction_2.timesOn)
           instructions.addData('button_next_instruction_2.timesOff', button_next_instruction_2.timesOff)
        else:
           instructions.addData('button_next_instruction_2.timesOn', "")
           instructions.addData('button_next_instruction_2.timesOff', "")
        instructions.addData('button_previous_instruction_2.numClicks', button_previous_instruction_2.numClicks)
        if button_previous_instruction_2.numClicks:
           instructions.addData('button_previous_instruction_2.timesOn', button_previous_instruction_2.timesOn)
           instructions.addData('button_previous_instruction_2.timesOff', button_previous_instruction_2.timesOff)
        else:
           instructions.addData('button_previous_instruction_2.timesOn', "")
           instructions.addData('button_previous_instruction_2.timesOff', "")
        # check responses
        if key_resp_skip_instructions_2.keys in ['', [], None]:  # No response was made
            key_resp_skip_instructions_2.keys = None
        instructions.addData('key_resp_skip_instructions_2.keys',key_resp_skip_instructions_2.keys)
        if key_resp_skip_instructions_2.keys != None:  # we had a response
            instructions.addData('key_resp_skip_instructions_2.rt', key_resp_skip_instructions_2.rt)
            instructions.addData('key_resp_skip_instructions_2.duration', key_resp_skip_instructions_2.duration)
        # the Routine "INSTRUCTIONS" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 1.0 repeats of 'instructions'
    
    
    # --- Prepare to start Routine "FIJACION_CENTRAL" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('FIJACION_CENTRAL.started', globalClock.getTime())
    # keep track of which components have finished
    FIJACION_CENTRALComponents = [polygon, gaze, GP_logs_2, GP_logs_3]
    for thisComponent in FIJACION_CENTRALComponents:
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
    
    # --- Run Routine "FIJACION_CENTRAL" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from end_of_routine
        if t>central_fixation_routine_time:
            continueRoutine = False
        # Run 'Each Frame' code from mouse_as_gaze
        gaze_position = mouse.getPosition()
        #logs_coordenadas_mirada.setText(f'{gaze_position[0]:.2f},{gaze_position[1]:.2f}')
        
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
        
        # *GP_logs_2* updates
        
        # if GP_logs_2 is starting this frame...
        if GP_logs_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            GP_logs_2.frameNStart = frameN  # exact frame index
            GP_logs_2.tStart = t  # local t and not account for scr refresh
            GP_logs_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(GP_logs_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            GP_logs_2.status = STARTED
            GP_logs_2.setAutoDraw(True)
        
        # if GP_logs_2 is active this frame...
        if GP_logs_2.status == STARTED:
            # update params
            pass
        
        # *GP_logs_3* updates
        
        # if GP_logs_3 is starting this frame...
        if GP_logs_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            GP_logs_3.frameNStart = frameN  # exact frame index
            GP_logs_3.tStart = t  # local t and not account for scr refresh
            GP_logs_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(GP_logs_3, 'tStartRefresh')  # time at next scr refresh
            # update status
            GP_logs_3.status = STARTED
            GP_logs_3.setAutoDraw(True)
        
        # if GP_logs_3 is active this frame...
        if GP_logs_3.status == STARTED:
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
        for thisComponent in FIJACION_CENTRALComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "FIJACION_CENTRAL" ---
    for thisComponent in FIJACION_CENTRALComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('FIJACION_CENTRAL.stopped', globalClock.getTime())
    # the Routine "FIJACION_CENTRAL" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "DESCANSO" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('DESCANSO.started', globalClock.getTime())
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # keep track of which components have finished
    DESCANSOComponents = [text_countdown, texto_descanso, key_resp]
    for thisComponent in DESCANSOComponents:
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
    
    # --- Run Routine "DESCANSO" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_countdown* updates
        
        # if text_countdown is starting this frame...
        if text_countdown.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_countdown.frameNStart = frameN  # exact frame index
            text_countdown.tStart = t  # local t and not account for scr refresh
            text_countdown.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_countdown, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_countdown.started')
            # update status
            text_countdown.status = STARTED
            text_countdown.setAutoDraw(True)
        
        # if text_countdown is active this frame...
        if text_countdown.status == STARTED:
            # update params
            text_countdown.setText(str(tiempo_descanso-int(t))
            , log=False)
        
        # if text_countdown is stopping this frame...
        if text_countdown.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_countdown.tStartRefresh + tiempo_descanso-frameTolerance:
                # keep track of stop time/frame for later
                text_countdown.tStop = t  # not accounting for scr refresh
                text_countdown.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_countdown.stopped')
                # update status
                text_countdown.status = FINISHED
                text_countdown.setAutoDraw(False)
        
        # *texto_descanso* updates
        
        # if texto_descanso is starting this frame...
        if texto_descanso.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            texto_descanso.frameNStart = frameN  # exact frame index
            texto_descanso.tStart = t  # local t and not account for scr refresh
            texto_descanso.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(texto_descanso, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'texto_descanso.started')
            # update status
            texto_descanso.status = STARTED
            texto_descanso.setAutoDraw(True)
        
        # if texto_descanso is active this frame...
        if texto_descanso.status == STARTED:
            # update params
            pass
        
        # if texto_descanso is stopping this frame...
        if texto_descanso.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > texto_descanso.tStartRefresh + tiempo_descanso-frameTolerance:
                # keep track of stop time/frame for later
                texto_descanso.tStop = t  # not accounting for scr refresh
                texto_descanso.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'texto_descanso.stopped')
                # update status
                texto_descanso.status = FINISHED
                texto_descanso.setAutoDraw(False)
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp is stopping this frame...
        if key_resp.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp.tStartRefresh + tiempo_descanso-frameTolerance:
                # keep track of stop time/frame for later
                key_resp.tStop = t  # not accounting for scr refresh
                key_resp.frameNStop = frameN  # exact frame index
                # update status
                key_resp.status = FINISHED
                key_resp.status = FINISHED
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
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
        for thisComponent in DESCANSOComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "DESCANSO" ---
    for thisComponent in DESCANSOComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('DESCANSO.stopped', globalClock.getTime())
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "DESCANSO" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "SACADAS" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('SACADAS.started', globalClock.getTime())
    # Run 'Begin Routine' code from positions
    stim_time = 2
    # keep track of which components have finished
    SACADASComponents = [sacade_stimuli]
    for thisComponent in SACADASComponents:
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
    
    # --- Run Routine "SACADAS" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from end_of_routine_2
        if t>sacade_routine_time:
            continueRoutine = False
        # Run 'Each Frame' code from positions
        # secuencia de movimientos
        step = 2
        if t>stim_time:
            actual_position_index+=1
            print(f'{t}>{stim_time} - New index i = {actual_position_index}')
            stim_time += step
            print(f"New position: {positions_list[actual_position_index]}")
        
        # *sacade_stimuli* updates
        
        # if sacade_stimuli is starting this frame...
        if sacade_stimuli.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            sacade_stimuli.frameNStart = frameN  # exact frame index
            sacade_stimuli.tStart = t  # local t and not account for scr refresh
            sacade_stimuli.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(sacade_stimuli, 'tStartRefresh')  # time at next scr refresh
            # update status
            sacade_stimuli.status = STARTED
            sacade_stimuli.setAutoDraw(True)
        
        # if sacade_stimuli is active this frame...
        if sacade_stimuli.status == STARTED:
            # update params
            sacade_stimuli.setPos([positions_list[actual_position_index]], log=False)
        
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
        for thisComponent in SACADASComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "SACADAS" ---
    for thisComponent in SACADASComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('SACADAS.stopped', globalClock.getTime())
    # the Routine "SACADAS" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "DESCANSO" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('DESCANSO.started', globalClock.getTime())
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # keep track of which components have finished
    DESCANSOComponents = [text_countdown, texto_descanso, key_resp]
    for thisComponent in DESCANSOComponents:
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
    
    # --- Run Routine "DESCANSO" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_countdown* updates
        
        # if text_countdown is starting this frame...
        if text_countdown.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_countdown.frameNStart = frameN  # exact frame index
            text_countdown.tStart = t  # local t and not account for scr refresh
            text_countdown.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_countdown, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_countdown.started')
            # update status
            text_countdown.status = STARTED
            text_countdown.setAutoDraw(True)
        
        # if text_countdown is active this frame...
        if text_countdown.status == STARTED:
            # update params
            text_countdown.setText(str(tiempo_descanso-int(t))
            , log=False)
        
        # if text_countdown is stopping this frame...
        if text_countdown.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_countdown.tStartRefresh + tiempo_descanso-frameTolerance:
                # keep track of stop time/frame for later
                text_countdown.tStop = t  # not accounting for scr refresh
                text_countdown.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_countdown.stopped')
                # update status
                text_countdown.status = FINISHED
                text_countdown.setAutoDraw(False)
        
        # *texto_descanso* updates
        
        # if texto_descanso is starting this frame...
        if texto_descanso.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            texto_descanso.frameNStart = frameN  # exact frame index
            texto_descanso.tStart = t  # local t and not account for scr refresh
            texto_descanso.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(texto_descanso, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'texto_descanso.started')
            # update status
            texto_descanso.status = STARTED
            texto_descanso.setAutoDraw(True)
        
        # if texto_descanso is active this frame...
        if texto_descanso.status == STARTED:
            # update params
            pass
        
        # if texto_descanso is stopping this frame...
        if texto_descanso.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > texto_descanso.tStartRefresh + tiempo_descanso-frameTolerance:
                # keep track of stop time/frame for later
                texto_descanso.tStop = t  # not accounting for scr refresh
                texto_descanso.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'texto_descanso.stopped')
                # update status
                texto_descanso.status = FINISHED
                texto_descanso.setAutoDraw(False)
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp is stopping this frame...
        if key_resp.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp.tStartRefresh + tiempo_descanso-frameTolerance:
                # keep track of stop time/frame for later
                key_resp.tStop = t  # not accounting for scr refresh
                key_resp.frameNStop = frameN  # exact frame index
                # update status
                key_resp.status = FINISHED
                key_resp.status = FINISHED
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
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
        for thisComponent in DESCANSOComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "DESCANSO" ---
    for thisComponent in DESCANSOComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('DESCANSO.stopped', globalClock.getTime())
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "DESCANSO" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "SEGUIMIENTO" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('SEGUIMIENTO.started', globalClock.getTime())
    # Run 'Begin Routine' code from movement_control_backend
    # Reiniciar el tiempo al comenzar la rutina
    #t0 = 0
    
    # keep track of which components have finished
    SEGUIMIENTOComponents = [moving_stimuli]
    for thisComponent in SEGUIMIENTOComponents:
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
    
    # --- Run Routine "SEGUIMIENTO" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *moving_stimuli* updates
        
        # if moving_stimuli is starting this frame...
        if moving_stimuli.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            moving_stimuli.frameNStart = frameN  # exact frame index
            moving_stimuli.tStart = t  # local t and not account for scr refresh
            moving_stimuli.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(moving_stimuli, 'tStartRefresh')  # time at next scr refresh
            # update status
            moving_stimuli.status = STARTED
            moving_stimuli.setAutoDraw(True)
        
        # if moving_stimuli is active this frame...
        if moving_stimuli.status == STARTED:
            # update params
            pass
        # Run 'Each Frame' code from movement_control_backend
        # Tiempo transcurrido
        #t = t - t0
        
        # Obtener el siguiente punto de la lista
        siguiente_punto = puntos[indice_punto]
        
        # Calcular la dirección hacia el siguiente punto
        direccion = [siguiente_punto[0] - moving_stimuli.pos[0], siguiente_punto[1] - moving_stimuli.pos[1]]
        
        # Normalizar la dirección
        norma = (direccion[0]**2 + direccion[1]**2) ** 0.5
        if norma > 0:
            direccion_normalizada = [direccion[0] / norma, direccion[1] / norma]
        else:
            direccion_normalizada = [0, 0]
        
        # Mover el estímulo hacia el siguiente punto
        moving_stimuli.pos = [moving_stimuli.pos[0] + direccion_normalizada[0] * velocidad,
                              moving_stimuli.pos[1] + direccion_normalizada[1] * velocidad]
        
        # Verificar si el estímulo ha alcanzado el siguiente punto
        if norma < 0.05:  # Tolerancia para considerar que ha llegado al punto
            indice_punto += 1
            if indice_punto >= len(puntos):
                indice_punto = 0  # Reiniciar a la primera posición
        
        # Run 'Each Frame' code from end_of_routine_3
        if t>sacade_routine_time:
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
        for thisComponent in SEGUIMIENTOComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "SEGUIMIENTO" ---
    for thisComponent in SEGUIMIENTOComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('SEGUIMIENTO.stopped', globalClock.getTime())
    # the Routine "SEGUIMIENTO" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "DESCANSO" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('DESCANSO.started', globalClock.getTime())
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # keep track of which components have finished
    DESCANSOComponents = [text_countdown, texto_descanso, key_resp]
    for thisComponent in DESCANSOComponents:
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
    
    # --- Run Routine "DESCANSO" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_countdown* updates
        
        # if text_countdown is starting this frame...
        if text_countdown.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_countdown.frameNStart = frameN  # exact frame index
            text_countdown.tStart = t  # local t and not account for scr refresh
            text_countdown.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_countdown, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_countdown.started')
            # update status
            text_countdown.status = STARTED
            text_countdown.setAutoDraw(True)
        
        # if text_countdown is active this frame...
        if text_countdown.status == STARTED:
            # update params
            text_countdown.setText(str(tiempo_descanso-int(t))
            , log=False)
        
        # if text_countdown is stopping this frame...
        if text_countdown.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_countdown.tStartRefresh + tiempo_descanso-frameTolerance:
                # keep track of stop time/frame for later
                text_countdown.tStop = t  # not accounting for scr refresh
                text_countdown.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_countdown.stopped')
                # update status
                text_countdown.status = FINISHED
                text_countdown.setAutoDraw(False)
        
        # *texto_descanso* updates
        
        # if texto_descanso is starting this frame...
        if texto_descanso.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            texto_descanso.frameNStart = frameN  # exact frame index
            texto_descanso.tStart = t  # local t and not account for scr refresh
            texto_descanso.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(texto_descanso, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'texto_descanso.started')
            # update status
            texto_descanso.status = STARTED
            texto_descanso.setAutoDraw(True)
        
        # if texto_descanso is active this frame...
        if texto_descanso.status == STARTED:
            # update params
            pass
        
        # if texto_descanso is stopping this frame...
        if texto_descanso.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > texto_descanso.tStartRefresh + tiempo_descanso-frameTolerance:
                # keep track of stop time/frame for later
                texto_descanso.tStop = t  # not accounting for scr refresh
                texto_descanso.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'texto_descanso.stopped')
                # update status
                texto_descanso.status = FINISHED
                texto_descanso.setAutoDraw(False)
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp is stopping this frame...
        if key_resp.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp.tStartRefresh + tiempo_descanso-frameTolerance:
                # keep track of stop time/frame for later
                key_resp.tStop = t  # not accounting for scr refresh
                key_resp.frameNStop = frameN  # exact frame index
                # update status
                key_resp.status = FINISHED
                key_resp.status = FINISHED
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
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
        for thisComponent in DESCANSOComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "DESCANSO" ---
    for thisComponent in DESCANSOComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('DESCANSO.stopped', globalClock.getTime())
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "DESCANSO" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "LECTURA" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('LECTURA.started', globalClock.getTime())
    # Run 'Begin Routine' code from text_config
    # Tamaño de la letra
    text_lectura.height = 0.05
    text_lectura.alignHoriz = 'left'  
    text_lectura.color = 'white'  
    
    # keep track of which components have finished
    LECTURAComponents = [text_lectura]
    for thisComponent in LECTURAComponents:
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
    
    # --- Run Routine "LECTURA" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from end_of_routine_4
        if t>sacade_routine_time:
            continueRoutine = False
        
        # *text_lectura* updates
        
        # if text_lectura is starting this frame...
        if text_lectura.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_lectura.frameNStart = frameN  # exact frame index
            text_lectura.tStart = t  # local t and not account for scr refresh
            text_lectura.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_lectura, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_lectura.started')
            # update status
            text_lectura.status = STARTED
            text_lectura.setAutoDraw(True)
        
        # if text_lectura is active this frame...
        if text_lectura.status == STARTED:
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
        for thisComponent in LECTURAComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "LECTURA" ---
    for thisComponent in LECTURAComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('LECTURA.stopped', globalClock.getTime())
    # the Routine "LECTURA" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
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
