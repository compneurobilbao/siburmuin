#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on abril 24, 2024, at 10:08
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

# Run 'Before Experiment' code from code_5
instruction_no = 0
messages_instructions = ["Bienvenido/a al Test de Percepción de Velocidad y Aceleración. En este test, observarás varios estímulos en movimiento y se te pedirá que realices ciertas tareas relacionadas con la percepción de su trayectoria, velocidad y aceleración.", 
"En primer lugar, vas a ver un estímulo (una circunferencia) que se mueve de izquierda a derecha y atraviesa una pared o un túnel. Tu función en el test es tratar de predecir cuando la circunferencia (estímulo) va a salir por la parte derecha del túnel. Debes pulsar la barra espaciadora cuando creas que se va a dar el momento.", 
"Al pulsar la barra espaciadora, el túnel se va a poner transparente y la bola inicial va a aparecer (dentro o fuera del tunel) para que veas lo cerca que te has quedado de acertar.",
"Comenzamos con una prueba antes de registrar los resultados.",
"Pulsa la barra espaciadora cuando estes preparado/a. Si tienes dudas, ¡pregunta!."]
# Run 'Before Experiment' code from code
referencia_tierra = 0.0# -0.2 para partir de la base de la pantalla. 0.2 parte superior de la pantalla.
stim_size = 0.055
#anchura_tunel = 100 # % del area máxima del tunel
stop = False
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'test_percepcion_velocidad_aceleracion_y_trayectoria'  # from the Builder filename that created this script
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
        originPath='C:\\Users\\akoun\\Desktop\\Biocruces\\siburmuin\\src\\psychopy_dynamic_tests\\[VR]test_percepcion_velocidad_aceleracion_y_trayectoria\\test_percepcion_velocidad_aceleracion_y_trayectoria_lastrun.py',
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
            size=[1440, 900], fullscr=True, screen=0,
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
    
    # --- Initialize components for Routine "instructions" ---
    logo_bio = visual.ImageStim(
        win=win,
        name='logo_bio', 
        image='images/BIOBIZKAIA_horizontal_CMYK.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.45, 0.35), size=(0.4, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    logo_compneurolab = visual.ImageStim(
        win=win,
        name='logo_compneurolab', 
        image='images/compneuro_horizontal.png', mask=None, anchor='center',
        ori=0.0, pos=(0.45, 0.35), size=(0.6, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    text_title = visual.TextStim(win=win, name='text_title',
        text='TEST DE PERCEPCIÓN DE VELOCIDAD, ACELERACIÓN Y TRAYECTORIA',
        font='Open Sans',
        pos=(0, 0.1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_instructions = visual.TextStim(win=win, name='text_instructions',
        text=None,
        font='Open Sans',
        pos=(0, -0.20), height=0.035, wrapWidth=1.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    # Run 'Begin Experiment' code from code_5
    #text_instructions.alignHoriz='left'
    #text_instructions.wrapWidth=1.0
    instruction_no = 0
    button_next_instruction = visual.ButtonStim(win, 
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
        name='button_next_instruction',
        depth=-5
    )
    button_next_instruction.buttonClock = core.Clock()
    button_previous_instruction = visual.ButtonStim(win, 
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
        name='button_previous_instruction',
        depth=-6
    )
    button_previous_instruction.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "wait" ---
    tunel_2 = visual.Rect(
        win=win, name='tunel_2',
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    polygon = visual.ShapeStim(
        win=win, name='polygon',
        size=(stim_size, stim_size), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    ground = visual.Rect(
        win=win, name='ground',
        width=(2, 1)[0], height=(2, 1)[1],
        ori=0.0, pos=(0, referencia_tierra-(1/2)), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    
    # --- Initialize components for Routine "start" ---
    stim = visual.ShapeStim(
        win=win, name='stim',
        size=(stim_size, stim_size), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    tunel = visual.Rect(
        win=win, name='tunel',
        width=[1.0, 1.0][0], height=[1.0, 1.0][1],
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    ground_2 = visual.Rect(
        win=win, name='ground_2',
        width=(2, 1)[0], height=(2, 1)[1],
        ori=0.0, pos=(0, referencia_tierra-(1/2)), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    text = visual.TextStim(win=win, name='text',
        text=None,
        font='Open Sans',
        pos=(0.25, 0.25), height=0.035, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    # Run 'Begin Experiment' code from code
    #velocidad = 1/10 # TODO: algoritmo cálculo de velocidad para cada iteración
    tunel.opacity = 1
    
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
    
    # --- Prepare to start Routine "instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instructions.started', globalClock.getTime())
    # reset button_next_instruction to account for continued clicks & clear times on/off
    button_next_instruction.reset()
    # reset button_previous_instruction to account for continued clicks & clear times on/off
    button_previous_instruction.reset()
    # keep track of which components have finished
    instructionsComponents = [logo_bio, logo_compneurolab, text_title, text_instructions, button_next_instruction, button_previous_instruction]
    for thisComponent in instructionsComponents:
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
    
    # --- Run Routine "instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *logo_bio* updates
        
        # if logo_bio is starting this frame...
        if logo_bio.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            logo_bio.frameNStart = frameN  # exact frame index
            logo_bio.tStart = t  # local t and not account for scr refresh
            logo_bio.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(logo_bio, 'tStartRefresh')  # time at next scr refresh
            # update status
            logo_bio.status = STARTED
            logo_bio.setAutoDraw(True)
        
        # if logo_bio is active this frame...
        if logo_bio.status == STARTED:
            # update params
            pass
        
        # *logo_compneurolab* updates
        
        # if logo_compneurolab is starting this frame...
        if logo_compneurolab.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            logo_compneurolab.frameNStart = frameN  # exact frame index
            logo_compneurolab.tStart = t  # local t and not account for scr refresh
            logo_compneurolab.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(logo_compneurolab, 'tStartRefresh')  # time at next scr refresh
            # update status
            logo_compneurolab.status = STARTED
            logo_compneurolab.setAutoDraw(True)
        
        # if logo_compneurolab is active this frame...
        if logo_compneurolab.status == STARTED:
            # update params
            pass
        
        # *text_title* updates
        
        # if text_title is starting this frame...
        if text_title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_title.frameNStart = frameN  # exact frame index
            text_title.tStart = t  # local t and not account for scr refresh
            text_title.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_title, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_title.status = STARTED
            text_title.setAutoDraw(True)
        
        # if text_title is active this frame...
        if text_title.status == STARTED:
            # update params
            pass
        
        # *text_instructions* updates
        
        # if text_instructions is starting this frame...
        if text_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instructions.frameNStart = frameN  # exact frame index
            text_instructions.tStart = t  # local t and not account for scr refresh
            text_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instructions, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_instructions.status = STARTED
            text_instructions.setAutoDraw(True)
        
        # if text_instructions is active this frame...
        if text_instructions.status == STARTED:
            # update params
            text_instructions.setText('', log=False)
        # Run 'Each Frame' code from code_5
        text_instructions.text = messages_instructions[instruction_no]
            
        if instruction_no == (len(messages_instructions) - 1):
            button_next_instruction.opacity = 0
            #button_next_instruction.status = PAUSED
        else:
            button_next_instruction.opacity = 1.0
            #button_next_instruction.status = STARTED
        
        if instruction_no == 0:
            button_previous_instruction.opacity = 0
            #button_previous_instruction.status = PAUSED
        else:
            button_previous_instruction.opacity = 1.0
            #button_previous_instruction.status = STARTED
        
        ###################################################
        ####________________EVENTS_____________________####
        ###################################################
        
        keys = event.getKeys()  # Cada llamada al buffer lo vaciía (teoricamente)
        
        if 'space' in keys:
            continueRoutine = False
        # *button_next_instruction* updates
        
        # if button_next_instruction is starting this frame...
        if button_next_instruction.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            button_next_instruction.frameNStart = frameN  # exact frame index
            button_next_instruction.tStart = t  # local t and not account for scr refresh
            button_next_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(button_next_instruction, 'tStartRefresh')  # time at next scr refresh
            # update status
            button_next_instruction.status = STARTED
            button_next_instruction.setAutoDraw(True)
        
        # if button_next_instruction is active this frame...
        if button_next_instruction.status == STARTED:
            # update params
            pass
            # check whether button_next_instruction has been pressed
            if button_next_instruction.isClicked:
                if not button_next_instruction.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    button_next_instruction.timesOn.append(button_next_instruction.buttonClock.getTime())
                    button_next_instruction.timesOff.append(button_next_instruction.buttonClock.getTime())
                elif len(button_next_instruction.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    button_next_instruction.timesOff[-1] = button_next_instruction.buttonClock.getTime()
                if not button_next_instruction.wasClicked:
                    # run callback code when button_next_instruction is clicked
                    if instruction_no < len(messages_instructions)-1:
                        instruction_no+=1
        # take note of whether button_next_instruction was clicked, so that next frame we know if clicks are new
        button_next_instruction.wasClicked = button_next_instruction.isClicked and button_next_instruction.status == STARTED
        # *button_previous_instruction* updates
        
        # if button_previous_instruction is starting this frame...
        if button_previous_instruction.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            button_previous_instruction.frameNStart = frameN  # exact frame index
            button_previous_instruction.tStart = t  # local t and not account for scr refresh
            button_previous_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(button_previous_instruction, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'button_previous_instruction.started')
            # update status
            button_previous_instruction.status = STARTED
            button_previous_instruction.setAutoDraw(True)
        
        # if button_previous_instruction is active this frame...
        if button_previous_instruction.status == STARTED:
            # update params
            pass
            # check whether button_previous_instruction has been pressed
            if button_previous_instruction.isClicked:
                if not button_previous_instruction.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    button_previous_instruction.timesOn.append(button_previous_instruction.buttonClock.getTime())
                    button_previous_instruction.timesOff.append(button_previous_instruction.buttonClock.getTime())
                elif len(button_previous_instruction.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    button_previous_instruction.timesOff[-1] = button_previous_instruction.buttonClock.getTime()
                if not button_previous_instruction.wasClicked:
                    # run callback code when button_previous_instruction is clicked
                    if 0 < instruction_no:
                        instruction_no-=1
        # take note of whether button_previous_instruction was clicked, so that next frame we know if clicks are new
        button_previous_instruction.wasClicked = button_previous_instruction.isClicked and button_previous_instruction.status == STARTED
        
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
        for thisComponent in instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instructions.stopped', globalClock.getTime())
    thisExp.addData('button_next_instruction.numClicks', button_next_instruction.numClicks)
    if button_next_instruction.numClicks:
       thisExp.addData('button_next_instruction.timesOn', button_next_instruction.timesOn)
       thisExp.addData('button_next_instruction.timesOff', button_next_instruction.timesOff)
    else:
       thisExp.addData('button_next_instruction.timesOn', "")
       thisExp.addData('button_next_instruction.timesOff', "")
    thisExp.addData('button_previous_instruction.numClicks', button_previous_instruction.numClicks)
    if button_previous_instruction.numClicks:
       thisExp.addData('button_previous_instruction.timesOn', button_previous_instruction.timesOn)
       thisExp.addData('button_previous_instruction.timesOff', button_previous_instruction.timesOff)
    else:
       thisExp.addData('button_previous_instruction.timesOn', "")
       thisExp.addData('button_previous_instruction.timesOff', "")
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_2 = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('specs.csv'),
        seed=None, name='trials_2')
    thisExp.addLoop(trials_2)  # add the loop to the experiment
    thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            globals()[paramName] = thisTrial_2[paramName]
    
    for thisTrial_2 in trials_2:
        currentLoop = trials_2
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
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                globals()[paramName] = thisTrial_2[paramName]
        
        # --- Prepare to start Routine "wait" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('wait.started', globalClock.getTime())
        # keep track of which components have finished
        waitComponents = [tunel_2, polygon, ground]
        for thisComponent in waitComponents:
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
        
        # --- Run Routine "wait" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *tunel_2* updates
            
            # if tunel_2 is starting this frame...
            if tunel_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                tunel_2.frameNStart = frameN  # exact frame index
                tunel_2.tStart = t  # local t and not account for scr refresh
                tunel_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(tunel_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'tunel_2.started')
                # update status
                tunel_2.status = STARTED
                tunel_2.setAutoDraw(True)
            
            # if tunel_2 is active this frame...
            if tunel_2.status == STARTED:
                # update params
                tunel_2.setPos((0+distancia_estimulo_tunel, referencia_tierra), log=False)
                tunel_2.setSize((anchura_tunel/100, 0.5), log=False)
            
            # *polygon* updates
            
            # if polygon is starting this frame...
            if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon.frameNStart = frameN  # exact frame index
                polygon.tStart = t  # local t and not account for scr refresh
                polygon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon.started')
                # update status
                polygon.status = STARTED
                polygon.setAutoDraw(True)
            
            # if polygon is active this frame...
            if polygon.status == STARTED:
                # update params
                polygon.setPos((-0.65, referencia_tierra+stim_size/2), log=False)
            
            # *ground* updates
            
            # if ground is starting this frame...
            if ground.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ground.frameNStart = frameN  # exact frame index
                ground.tStart = t  # local t and not account for scr refresh
                ground.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ground, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ground.started')
                # update status
                ground.status = STARTED
                ground.setAutoDraw(True)
            
            # if ground is active this frame...
            if ground.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from code_6
            ###################################################
            ####________________EVENTS_____________________####
            ###################################################
            
            keys = event.getKeys()  # Cada llamada al buffer lo vaciía (teoricamente)
            
            if 'space' in keys: # No solo se tiene que dar el evento, sino que se tiene que dar en la rutina. Si viene dado de la rutina anterior no cuenta.
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
            for thisComponent in waitComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "wait" ---
        for thisComponent in waitComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('wait.stopped', globalClock.getTime())
        # the Routine "wait" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "start" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('start.started', globalClock.getTime())
        # Run 'Begin Routine' code from code
        desplazamiento = 0
        stop = False
        tunel.opacity = 1
        # keep track of which components have finished
        startComponents = [stim, tunel, ground_2, text]
        for thisComponent in startComponents:
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
        
        # --- Run Routine "start" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
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
                stim.setPos((-0.65 + desplazamiento, referencia_tierra+stim_size/2), log=False)
            
            # *tunel* updates
            
            # if tunel is starting this frame...
            if tunel.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                tunel.frameNStart = frameN  # exact frame index
                tunel.tStart = t  # local t and not account for scr refresh
                tunel.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(tunel, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'tunel.started')
                # update status
                tunel.status = STARTED
                tunel.setAutoDraw(True)
            
            # if tunel is active this frame...
            if tunel.status == STARTED:
                # update params
                tunel.setPos((0+distancia_estimulo_tunel, referencia_tierra), log=False)
                tunel.setSize((anchura_tunel/100, 0.5), log=False)
            
            # *ground_2* updates
            
            # if ground_2 is starting this frame...
            if ground_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ground_2.frameNStart = frameN  # exact frame index
                ground_2.tStart = t  # local t and not account for scr refresh
                ground_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ground_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ground_2.started')
                # update status
                ground_2.status = STARTED
                ground_2.setAutoDraw(True)
            
            # if ground_2 is active this frame...
            if ground_2.status == STARTED:
                # update params
                pass
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                text.setText('', log=False)
            # Run 'Each Frame' code from code
            if stop:
                pass
            else:
                desplazamiento = t*velocidad_2
            
            # Obtiene la posición en X de 'stim'
            stim_x = stim.pos[0]
            
            # Obtiene la posición en X del centro de 'box' y calcula la posición del borde derecho
            box_right_edge_x = tunel.pos[0] + tunel.size[0] / 2
            
            # Calcula la distancia al borde derecho de 'box'
            distance_to_right_edge = box_right_edge_x - stim_x
            
            # Actualiza el componente de texto con la distancia al borde derecho
            text.setText(f"referencia_tierra: {referencia_tierra:.2f}\nDistancia: {distance_to_right_edge:.2f}\n Velocidad: {velocidad:.2f}\n Anchura tunel (%): {anchura_tunel:.2f}\n Distancia inicial: {distancia_estimulo_tunel:.2f}\n")
            
            tunel_right_edge_x = tunel.pos[0] + tunel.size[0] / 2
            
            # Hacer el estímulo invisible antes de salir por el túnel
            if stim_x > tunel_right_edge_x - 0.1 and not stop:
                stim.opacity = 0.0
            
            
            ####################################################
            ###################____EVENTS____###################
            ####################################################
            
            # Verifica si se ha presionado la tecla Space
            keys = event.getKeys()
            if 'space' in keys:
                if stop:    # Si ya se habia parado, el siguiente evento es para saltar de rutina
                    continueRoutine = False
                else:       # Primera vez que se pulsa la barra: parar estimulo y calcular errores
                    stop = True
                    #velocidad_2 = 0
                    tunel.opacity = 0.3
                    stim.opacity = 1.0
            
            
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
            for thisComponent in startComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "start" ---
        for thisComponent in startComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('start.stopped', globalClock.getTime())
        # the Routine "start" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_2'
    
    
    # set up handler to look after randomisation of conditions etc
    trials_3 = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('cube_speed_angles.csv'),
        seed=None, name='trials_3')
    thisExp.addLoop(trials_3)  # add the loop to the experiment
    thisTrial_3 = trials_3.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
    if thisTrial_3 != None:
        for paramName in thisTrial_3:
            globals()[paramName] = thisTrial_3[paramName]
    
    for thisTrial_3 in trials_3:
        currentLoop = trials_3
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
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
        if thisTrial_3 != None:
            for paramName in thisTrial_3:
                globals()[paramName] = thisTrial_3[paramName]
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_3'
    
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=4.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('specs_tiro_parabolico.csv'),
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
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 4.0 repeats of 'trials'
    
    
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
