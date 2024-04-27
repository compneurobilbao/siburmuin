#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on abril 27, 2024, at 19:20
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
        originPath='C:\\Users\\akoun\\Desktop\\Biocruces\\siburmuin\\src\\psychopy_dynamic_tests\\magnocellular_stimuli\\magnocellular_stimuli_lastrun.py',
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
    
    # --- Initialize components for Routine "Magnocelular_test" ---
    # Run 'Begin Experiment' code from code
    from psychopy.iohub import launchHubServer
    io = launchHubServer()
    mouse = io.devices.mouse
    
    posicion_estimulo = (0,0)
    stim_x = 0
    stim_y = 0
    
    periphereal_region_diameter = 0.8
    periphereal_region = visual.ShapeStim(
        win=win, name='periphereal_region',
        size=[1.0, 1.0], vertices='circle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    foveal_region = visual.ShapeStim(
        win=win, name='foveal_region',
        size=(0.25, 0.25), vertices='circle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    logs = visual.TextStim(win=win, name='logs',
        text=None,
        font='Open Sans',
        pos=(-0.45, 0.45), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    logs_coordenadas_mirada = visual.TextStim(win=win, name='logs_coordenadas_mirada',
        text=None,
        font='Open Sans',
        pos=(-0.45, -0.45), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    logs_parametros_trial = visual.TextStim(win=win, name='logs_parametros_trial',
        text=None,
        font='Open Sans',
        pos=(0.45, 0.35), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    stim = visual.GratingStim(
        win=win, name='stim',
        tex='sin', mask='gauss', anchor='center',
        ori=1.0, pos=[0,0], size=1.0, sf=1.0, phase=0.5,
        color='white', colorSpace='rgb',
        opacity=0.7, contrast=1.0, blendmode='avg',
        texRes=512.0, interpolate=True, depth=-6.0)
    key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "break_5s_" ---
    text_countdown = visual.TextStim(win=win, name='text_countdown',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Parvocelular_test" ---
    # Run 'Begin Experiment' code from code_2
    from psychopy.iohub import launchHubServer
    io = launchHubServer()
    mouse = io.devices.mouse
    
    posicion_estimulo = (0,0)
    stim_x = 0
    stim_y = 0
    foveal_region_2 = visual.ShapeStim(
        win=win, name='foveal_region_2',
        size=(0.25, 0.25), vertices='circle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    logs_2 = visual.TextStim(win=win, name='logs_2',
        text=None,
        font='Open Sans',
        pos=(-0.45, 0.45), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    logs_coordenadas_mirada_2 = visual.TextStim(win=win, name='logs_coordenadas_mirada_2',
        text=None,
        font='Open Sans',
        pos=(-0.45, -0.45), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    logs_parametros_trial_2 = visual.TextStim(win=win, name='logs_parametros_trial_2',
        text=None,
        font='Open Sans',
        pos=(0.45, 0.35), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    stim_2 = visual.GratingStim(
        win=win, name='stim_2',
        tex='sin', mask='gauss', anchor='center',
        ori=1.0, pos=[0,0], size=1.0, sf=1.0, phase=0.5,
        color='white', colorSpace='rgb',
        opacity=0.7, contrast=1.0, blendmode='avg',
        texRes=512.0, interpolate=True, depth=-5.0)
    key_resp_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "break_5s_" ---
    text_countdown = visual.TextStim(win=win, name='text_countdown',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Magnocelular_test" ---
    # Run 'Begin Experiment' code from code
    from psychopy.iohub import launchHubServer
    io = launchHubServer()
    mouse = io.devices.mouse
    
    posicion_estimulo = (0,0)
    stim_x = 0
    stim_y = 0
    
    periphereal_region_diameter = 0.8
    periphereal_region = visual.ShapeStim(
        win=win, name='periphereal_region',
        size=[1.0, 1.0], vertices='circle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    foveal_region = visual.ShapeStim(
        win=win, name='foveal_region',
        size=(0.25, 0.25), vertices='circle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    logs = visual.TextStim(win=win, name='logs',
        text=None,
        font='Open Sans',
        pos=(-0.45, 0.45), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    logs_coordenadas_mirada = visual.TextStim(win=win, name='logs_coordenadas_mirada',
        text=None,
        font='Open Sans',
        pos=(-0.45, -0.45), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    logs_parametros_trial = visual.TextStim(win=win, name='logs_parametros_trial',
        text=None,
        font='Open Sans',
        pos=(0.45, 0.35), height=0.025, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    stim = visual.GratingStim(
        win=win, name='stim',
        tex='sin', mask='gauss', anchor='center',
        ori=1.0, pos=[0,0], size=1.0, sf=1.0, phase=0.5,
        color='white', colorSpace='rgb',
        opacity=0.7, contrast=1.0, blendmode='avg',
        texRes=512.0, interpolate=True, depth=-6.0)
    key_resp = keyboard.Keyboard()
    
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
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('settings.csv', selection='0:5'),
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
        
        # --- Prepare to start Routine "Magnocelular_test" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Magnocelular_test.started', globalClock.getTime())
        # Run 'Begin Routine' code from code
        import math
        import random
        
        def calcular_posicion_stim(diameter=1):
            radius = diameter/2
            
            # Generar un ángulo aleatorio
            theta = random.uniform(0, 2 * math.pi)
            
            # Calcular x e y segun el angulo
            stim_x = radius * math.cos(theta)
            stim_y = radius * math.sin(theta)
            
            return stim_x, stim_y
        
        ####################################################
        ###############____PARAMS CONFIG____################
        ####################################################
        posicion_estimulo = stim_x, stim_y = calcular_posicion_stim(periphereal_region_diameter)
        stim.sf = frecuencia_espacial
        stim.orientation = orientacion
        periphereal_region.setSize(periphereal_region_diameter)
        stim.setColor([color_r, color_g, color_b], colorSpace='rgb')
        stim.setContrast(contraste)
        stim.setPos((stim_x, stim_y))
        stim.setSize(tamanyo)
        stim.setOri(orientacion)
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # keep track of which components have finished
        Magnocelular_testComponents = [periphereal_region, foveal_region, logs, logs_coordenadas_mirada, logs_parametros_trial, stim, key_resp]
        for thisComponent in Magnocelular_testComponents:
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
        
        # --- Run Routine "Magnocelular_test" ---
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
                f"Intento: {intento}\n"
                f"Orientación: {orientacion:.2f}\n"
                f"Frecuencia Espacial: {frecuencia_espacial:.2f}\n"
                f"Frecuencia Temporal: {frecuencia_temporal:.3f}\n"
                f"Posicion Estimulo: ({posicion_estimulo[0]:.2f}, {posicion_estimulo[1]:.2f})\n"
                f"Contraste: {contraste:.2f}\n"
                f"Color: {color_r},{color_g},{color_b}\n"
                f"Tamaño: {tamanyo:.2f}"
            )
            ####################################################
            #################____SETTINGS____###################
            ####################################################
            
            stim.setPhase(frecuencia_temporal,'+')
            #stim.setPhase(0.01,'+')
            stim.draw()
            
            ####################################################
            ##########____GAZE VS REGION POSITION____###########
            ####################################################
            # Calcula la distancia del ratón al centro de foveal_region
            dist_from_center = ((gaze_position[0] - foveal_region.pos[0])**2 + (gaze_position[1] - foveal_region.pos[1])**2)**0.5
            
            # Comprueba si la distancia es menor que el radio de foveal_region
            if dist_from_center <= 0.25/2:#foveal_region.radius:
                logs.setText("El ratón está dentro de la circunferencia")
                # Aquí puedes hacer que el estímulo grating sea visible si es necesario
                  # o cualquier acción que quieras realizar
            else:
                logs.setText("El ratón está fuera de la circunferencia")
                # El ratón está fuera de la circunferencia
                # Aquí puedes hacer que el estímulo grating sea invisible si es necesario
                # o cualquier acción que quieras realizar
            
            ####################################################
            ##############____EVENTS & STATES____###############
            ####################################################
                
            # START/STOP: Verifica si se ha presionado la tecla
            keys = event.getKeys()
            if 'space' in keys:
                pass
                #stim_x, stim_y = calcular_posicion_stim(periphereal_region_diameter)
                
            
            # *periphereal_region* updates
            
            # if periphereal_region is starting this frame...
            if periphereal_region.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                periphereal_region.frameNStart = frameN  # exact frame index
                periphereal_region.tStart = t  # local t and not account for scr refresh
                periphereal_region.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(periphereal_region, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'periphereal_region.started')
                # update status
                periphereal_region.status = STARTED
                periphereal_region.setAutoDraw(True)
            
            # if periphereal_region is active this frame...
            if periphereal_region.status == STARTED:
                # update params
                pass
            
            # *foveal_region* updates
            
            # if foveal_region is starting this frame...
            if foveal_region.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                foveal_region.frameNStart = frameN  # exact frame index
                foveal_region.tStart = t  # local t and not account for scr refresh
                foveal_region.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(foveal_region, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'foveal_region.started')
                # update status
                foveal_region.status = STARTED
                foveal_region.setAutoDraw(True)
            
            # if foveal_region is active this frame...
            if foveal_region.status == STARTED:
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
            for thisComponent in Magnocelular_testComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Magnocelular_test" ---
        for thisComponent in Magnocelular_testComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Magnocelular_test.stopped', globalClock.getTime())
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        trials.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            trials.addData('key_resp.rt', key_resp.rt)
            trials.addData('key_resp.duration', key_resp.duration)
        # the Routine "Magnocelular_test" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials'
    
    
    # --- Prepare to start Routine "break_5s_" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('break_5s_.started', globalClock.getTime())
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # keep track of which components have finished
    break_5s_Components = [text_countdown, key_resp_3]
    for thisComponent in break_5s_Components:
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
    
    # --- Run Routine "break_5s_" ---
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
            text_countdown.setText(str(5-int(t)), log=False)
        
        # if text_countdown is stopping this frame...
        if text_countdown.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_countdown.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                text_countdown.tStop = t  # not accounting for scr refresh
                text_countdown.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_countdown.stopped')
                # update status
                text_countdown.status = FINISHED
                text_countdown.setAutoDraw(False)
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_3.started')
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
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
        for thisComponent in break_5s_Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_5s_" ---
    for thisComponent in break_5s_Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('break_5s_.stopped', globalClock.getTime())
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "break_5s_" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_2 = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('settings.csv', selection='5:10'),
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
        
        # --- Prepare to start Routine "Parvocelular_test" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Parvocelular_test.started', globalClock.getTime())
        # Run 'Begin Routine' code from code_2
        import math
        import random
        
        ####################################################
        ###############____PARAMS CONFIG____################
        ####################################################
        posicion_estimulo = stim_x, stim_y = 0,0
        stim_2.sf = frecuencia_espacial
        stim_2.orientation = orientacion
        stim_2.setColor([color_r,color_g,color_b], colorSpace='rgb')
        stim_2.setContrast(contraste)
        stim_2.setSize(tamanyo)
        stim_2.setOri(orientacion)
        key_resp_2.keys = []
        key_resp_2.rt = []
        _key_resp_2_allKeys = []
        # keep track of which components have finished
        Parvocelular_testComponents = [foveal_region_2, logs_2, logs_coordenadas_mirada_2, logs_parametros_trial_2, stim_2, key_resp_2]
        for thisComponent in Parvocelular_testComponents:
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
        
        # --- Run Routine "Parvocelular_test" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_2
            ####################################################
            ##############____ON SCREEN LOGS____################
            ####################################################
            gaze_position = mouse.getPosition()
            logs_coordenadas_mirada_2.setText(f'{gaze_position[0]:.2f},{gaze_position[1]:.2f}')
            logs_parametros_trial_2.setText(
                f"Intento: {intento}\n"
                f"Orientación: {orientacion:.2f}\n"
                f"Frecuencia Espacial: {frecuencia_espacial:.2f}\n"
                f"Frecuencia Temporal: {frecuencia_temporal:.3f}\n"
                f"Posicion Estimulo: ({posicion_estimulo[0]:.2f}, {posicion_estimulo[1]:.2f})\n"
                f"Contraste: {contraste:.2f}\n"
                f"Color: {color_r},{color_g},{color_b}\n"
                f"Tamaño: {tamanyo:.2f}"
            )
            
            ####################################################
            #################____SETTINGS____###################
            ####################################################
            
            stim_2.setPhase(frecuencia_temporal,'+')
            #stim_2.setPhase(0.01,'+')
            stim_2.draw()
            
            ####################################################
            ##########____GAZE VS REGION POSITION____###########
            ####################################################
            # Calcula la distancia del ratón al centro de foveal_region
            dist_from_center = ((gaze_position[0] - foveal_region_2.pos[0])**2 + (gaze_position[1] - foveal_region_2.pos[1])**2)**0.5
            
            # Comprueba si la distancia es menor que el radio de foveal_region
            if dist_from_center <= 0.25/2:#foveal_region.radius:
                logs_2.setText("El ratón está dentro de la circunferencia")
                # Aquí puedes hacer que el estímulo grating sea visible si es necesario
                  # o cualquier acción que quieras realizar
            else:
                logs_2.setText("El ratón está fuera de la circunferencia")
                # El ratón está fuera de la circunferencia
                # Aquí puedes hacer que el estímulo grating sea invisible si es necesario
                # o cualquier acción que quieras realizar
            
            ####################################################
            ##############____EVENTS & STATES____###############
            ####################################################
                
            # START/STOP: Verifica si se ha presionado la tecla
            keys = event.getKeys()
            if 'space' in keys:
                pass
                #stim_x, stim_y = calcular_posicion_stim(periphereal_region_diameter)
                
            
            # *foveal_region_2* updates
            
            # if foveal_region_2 is starting this frame...
            if foveal_region_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                foveal_region_2.frameNStart = frameN  # exact frame index
                foveal_region_2.tStart = t  # local t and not account for scr refresh
                foveal_region_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(foveal_region_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'foveal_region_2.started')
                # update status
                foveal_region_2.status = STARTED
                foveal_region_2.setAutoDraw(True)
            
            # if foveal_region_2 is active this frame...
            if foveal_region_2.status == STARTED:
                # update params
                pass
            
            # *logs_2* updates
            
            # if logs_2 is starting this frame...
            if logs_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_2.frameNStart = frameN  # exact frame index
                logs_2.tStart = t  # local t and not account for scr refresh
                logs_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'logs_2.started')
                # update status
                logs_2.status = STARTED
                logs_2.setAutoDraw(True)
            
            # if logs_2 is active this frame...
            if logs_2.status == STARTED:
                # update params
                pass
            
            # *logs_coordenadas_mirada_2* updates
            
            # if logs_coordenadas_mirada_2 is starting this frame...
            if logs_coordenadas_mirada_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_coordenadas_mirada_2.frameNStart = frameN  # exact frame index
                logs_coordenadas_mirada_2.tStart = t  # local t and not account for scr refresh
                logs_coordenadas_mirada_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_coordenadas_mirada_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'logs_coordenadas_mirada_2.started')
                # update status
                logs_coordenadas_mirada_2.status = STARTED
                logs_coordenadas_mirada_2.setAutoDraw(True)
            
            # if logs_coordenadas_mirada_2 is active this frame...
            if logs_coordenadas_mirada_2.status == STARTED:
                # update params
                pass
            
            # *logs_parametros_trial_2* updates
            
            # if logs_parametros_trial_2 is starting this frame...
            if logs_parametros_trial_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                logs_parametros_trial_2.frameNStart = frameN  # exact frame index
                logs_parametros_trial_2.tStart = t  # local t and not account for scr refresh
                logs_parametros_trial_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(logs_parametros_trial_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                logs_parametros_trial_2.status = STARTED
                logs_parametros_trial_2.setAutoDraw(True)
            
            # if logs_parametros_trial_2 is active this frame...
            if logs_parametros_trial_2.status == STARTED:
                # update params
                pass
            
            # *stim_2* updates
            
            # if stim_2 is starting this frame...
            if stim_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                stim_2.frameNStart = frameN  # exact frame index
                stim_2.tStart = t  # local t and not account for scr refresh
                stim_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stim_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                stim_2.status = STARTED
                stim_2.setAutoDraw(True)
            
            # if stim_2 is active this frame...
            if stim_2.status == STARTED:
                # update params
                stim_2.setPos((stim_x, stim_y), log=False)
            
            # *key_resp_2* updates
            
            # if key_resp_2 is starting this frame...
            if key_resp_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_2.frameNStart = frameN  # exact frame index
                key_resp_2.tStart = t  # local t and not account for scr refresh
                key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                key_resp_2.status = STARTED
                # keyboard checking is just starting
                key_resp_2.clock.reset()  # now t=0
            if key_resp_2.status == STARTED:
                theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_2_allKeys.extend(theseKeys)
                if len(_key_resp_2_allKeys):
                    key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                    key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                    key_resp_2.duration = _key_resp_2_allKeys[-1].duration
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
            for thisComponent in Parvocelular_testComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Parvocelular_test" ---
        for thisComponent in Parvocelular_testComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Parvocelular_test.stopped', globalClock.getTime())
        # check responses
        if key_resp_2.keys in ['', [], None]:  # No response was made
            key_resp_2.keys = None
        trials_2.addData('key_resp_2.keys',key_resp_2.keys)
        if key_resp_2.keys != None:  # we had a response
            trials_2.addData('key_resp_2.rt', key_resp_2.rt)
            trials_2.addData('key_resp_2.duration', key_resp_2.duration)
        # the Routine "Parvocelular_test" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_2'
    
    
    # --- Prepare to start Routine "break_5s_" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('break_5s_.started', globalClock.getTime())
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # keep track of which components have finished
    break_5s_Components = [text_countdown, key_resp_3]
    for thisComponent in break_5s_Components:
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
    
    # --- Run Routine "break_5s_" ---
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
            text_countdown.setText(str(5-int(t)), log=False)
        
        # if text_countdown is stopping this frame...
        if text_countdown.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_countdown.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                text_countdown.tStop = t  # not accounting for scr refresh
                text_countdown.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_countdown.stopped')
                # update status
                text_countdown.status = FINISHED
                text_countdown.setAutoDraw(False)
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_3.started')
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
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
        for thisComponent in break_5s_Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "break_5s_" ---
    for thisComponent in break_5s_Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('break_5s_.stopped', globalClock.getTime())
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "break_5s_" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_3 = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('settings.csv', selection='10:15'),
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
        
        # --- Prepare to start Routine "Magnocelular_test" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Magnocelular_test.started', globalClock.getTime())
        # Run 'Begin Routine' code from code
        import math
        import random
        
        def calcular_posicion_stim(diameter=1):
            radius = diameter/2
            
            # Generar un ángulo aleatorio
            theta = random.uniform(0, 2 * math.pi)
            
            # Calcular x e y segun el angulo
            stim_x = radius * math.cos(theta)
            stim_y = radius * math.sin(theta)
            
            return stim_x, stim_y
        
        ####################################################
        ###############____PARAMS CONFIG____################
        ####################################################
        posicion_estimulo = stim_x, stim_y = calcular_posicion_stim(periphereal_region_diameter)
        stim.sf = frecuencia_espacial
        stim.orientation = orientacion
        periphereal_region.setSize(periphereal_region_diameter)
        stim.setColor([color_r, color_g, color_b], colorSpace='rgb')
        stim.setContrast(contraste)
        stim.setPos((stim_x, stim_y))
        stim.setSize(tamanyo)
        stim.setOri(orientacion)
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # keep track of which components have finished
        Magnocelular_testComponents = [periphereal_region, foveal_region, logs, logs_coordenadas_mirada, logs_parametros_trial, stim, key_resp]
        for thisComponent in Magnocelular_testComponents:
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
        
        # --- Run Routine "Magnocelular_test" ---
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
                f"Intento: {intento}\n"
                f"Orientación: {orientacion:.2f}\n"
                f"Frecuencia Espacial: {frecuencia_espacial:.2f}\n"
                f"Frecuencia Temporal: {frecuencia_temporal:.3f}\n"
                f"Posicion Estimulo: ({posicion_estimulo[0]:.2f}, {posicion_estimulo[1]:.2f})\n"
                f"Contraste: {contraste:.2f}\n"
                f"Color: {color_r},{color_g},{color_b}\n"
                f"Tamaño: {tamanyo:.2f}"
            )
            ####################################################
            #################____SETTINGS____###################
            ####################################################
            
            stim.setPhase(frecuencia_temporal,'+')
            #stim.setPhase(0.01,'+')
            stim.draw()
            
            ####################################################
            ##########____GAZE VS REGION POSITION____###########
            ####################################################
            # Calcula la distancia del ratón al centro de foveal_region
            dist_from_center = ((gaze_position[0] - foveal_region.pos[0])**2 + (gaze_position[1] - foveal_region.pos[1])**2)**0.5
            
            # Comprueba si la distancia es menor que el radio de foveal_region
            if dist_from_center <= 0.25/2:#foveal_region.radius:
                logs.setText("El ratón está dentro de la circunferencia")
                # Aquí puedes hacer que el estímulo grating sea visible si es necesario
                  # o cualquier acción que quieras realizar
            else:
                logs.setText("El ratón está fuera de la circunferencia")
                # El ratón está fuera de la circunferencia
                # Aquí puedes hacer que el estímulo grating sea invisible si es necesario
                # o cualquier acción que quieras realizar
            
            ####################################################
            ##############____EVENTS & STATES____###############
            ####################################################
                
            # START/STOP: Verifica si se ha presionado la tecla
            keys = event.getKeys()
            if 'space' in keys:
                pass
                #stim_x, stim_y = calcular_posicion_stim(periphereal_region_diameter)
                
            
            # *periphereal_region* updates
            
            # if periphereal_region is starting this frame...
            if periphereal_region.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                periphereal_region.frameNStart = frameN  # exact frame index
                periphereal_region.tStart = t  # local t and not account for scr refresh
                periphereal_region.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(periphereal_region, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'periphereal_region.started')
                # update status
                periphereal_region.status = STARTED
                periphereal_region.setAutoDraw(True)
            
            # if periphereal_region is active this frame...
            if periphereal_region.status == STARTED:
                # update params
                pass
            
            # *foveal_region* updates
            
            # if foveal_region is starting this frame...
            if foveal_region.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                foveal_region.frameNStart = frameN  # exact frame index
                foveal_region.tStart = t  # local t and not account for scr refresh
                foveal_region.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(foveal_region, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'foveal_region.started')
                # update status
                foveal_region.status = STARTED
                foveal_region.setAutoDraw(True)
            
            # if foveal_region is active this frame...
            if foveal_region.status == STARTED:
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
            for thisComponent in Magnocelular_testComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Magnocelular_test" ---
        for thisComponent in Magnocelular_testComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Magnocelular_test.stopped', globalClock.getTime())
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        trials_3.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            trials_3.addData('key_resp.rt', key_resp.rt)
            trials_3.addData('key_resp.duration', key_resp.duration)
        # the Routine "Magnocelular_test" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_3'
    
    
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
    expInfo = showExpInfoDlg(expInfo=expInfo)
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
