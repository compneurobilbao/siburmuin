﻿#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on febrero 21, 2024, at 15:19
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
time = 1
dots_no_1 = 3500
dots_no_2 = 3500
dots_no_3 = 3500
dots_no_4 = 3500
dots_no_5 = 3500
dots_no_6 = 3500
dots_no_7 = 3500
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'gollin_figure_test_v2'  # from the Builder filename that created this script
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
        originPath='C:\\Users\\akoun\\Desktop\\Biocruces\\psychopy\\Gollin_figure_test\\gollin_figure_test_v2_lastrun.py',
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
            monitor='testMonitor', color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [1.0000, 1.0000, 1.0000]
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
    
    # --- Initialize components for Routine "logs" ---
    # Run 'Begin Experiment' code from code_2
    iteration_no = 0
    
    
    # --- Initialize components for Routine "trial_1" ---
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    dots_1 = visual.DotStim(
        win=win, name='dots_1',
        nDots=dots_no_1, dotSize=20.0,
        speed=0.0, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=1.0, fieldAnchor='center', fieldShape='circle',
        signalDots='same', noiseDots='position',dotLife=-1.0,
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None,
        depth=-1.0)
    dots_2 = visual.DotStim(
        win=win, name='dots_2',
        nDots=dots_no_2, dotSize=20.0,
        speed=0.0, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=1.0, fieldAnchor='center', fieldShape='circle',
        signalDots='same', noiseDots='position',dotLife=-1.0,
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None,
        depth=-2.0)
    dots_3 = visual.DotStim(
        win=win, name='dots_3',
        nDots=dots_no_3, dotSize=20.0,
        speed=0.0, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=1.0, fieldAnchor='center', fieldShape='circle',
        signalDots='same', noiseDots='position',dotLife=-1.0,
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None,
        depth=-3.0)
    dots_4 = visual.DotStim(
        win=win, name='dots_4',
        nDots=dots_no_4, dotSize=20.0,
        speed=0.0, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=1.0, fieldAnchor='center', fieldShape='circle',
        signalDots='same', noiseDots='position',dotLife=-1.0,
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None,
        depth=-4.0)
    dots_5 = visual.DotStim(
        win=win, name='dots_5',
        nDots=dots_no_5, dotSize=20.0,
        speed=0.0, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=1.0, fieldAnchor='center', fieldShape='circle',
        signalDots='same', noiseDots='position',dotLife=-1.0,
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None,
        depth=-5.0)
    dots_6 = visual.DotStim(
        win=win, name='dots_6',
        nDots=dots_no_6, dotSize=20.0,
        speed=0.0, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=1.0, fieldAnchor='center', fieldShape='circle',
        signalDots='same', noiseDots='position',dotLife=-1.0,
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None,
        depth=-6.0)
    dots_7 = visual.DotStim(
        win=win, name='dots_7',
        nDots=dots_no_7, dotSize=20.0,
        speed=0.0, dir=0.0, coherence=1.0,
        fieldPos=(0.0, 0.0), fieldSize=1.0, fieldAnchor='center', fieldShape='circle',
        signalDots='same', noiseDots='position',dotLife=-1.0,
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None,
        depth=-7.0)
    key_resp_8 = keyboard.Keyboard()
    # Run 'Begin Experiment' code from code
    estimulos = [dots_1, dots_2, dots_3, dots_4, dots_5, dots_6, dots_7]
    
    
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
    trials = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('images/images.csv'),
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
        
        # --- Prepare to start Routine "logs" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('logs.started', globalClock.getTime())
        # Run 'Begin Routine' code from code_2
        image_iteration = 0
        print("------------------------------------")
        print("Loading \"", file ,"\" file")
        print("Iteration number "+ str(iteration_no) + ".")
        print("------------------------------------")
        # keep track of which components have finished
        logsComponents = []
        for thisComponent in logsComponents:
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
        
        # --- Run Routine "logs" ---
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
            for thisComponent in logsComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "logs" ---
        for thisComponent in logsComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('logs.stopped', globalClock.getTime())
        # Run 'End Routine' code from code_2
        iteration_no += 1
        # the Routine "logs" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        filter_removal = data.TrialHandler(nReps=8.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='filter_removal')
        thisExp.addLoop(filter_removal)  # add the loop to the experiment
        thisFilter_removal = filter_removal.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisFilter_removal.rgb)
        if thisFilter_removal != None:
            for paramName in thisFilter_removal:
                globals()[paramName] = thisFilter_removal[paramName]
        
        for thisFilter_removal in filter_removal:
            currentLoop = filter_removal
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
            # abbreviate parameter names if possible (e.g. rgb = thisFilter_removal.rgb)
            if thisFilter_removal != None:
                for paramName in thisFilter_removal:
                    globals()[paramName] = thisFilter_removal[paramName]
            
            # --- Prepare to start Routine "trial_1" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial_1.started', globalClock.getTime())
            image.setImage(file)
            key_resp_8.keys = []
            key_resp_8.rt = []
            _key_resp_8_allKeys = []
            # Run 'Begin Routine' code from code
            
            
            if image_iteration == 0:
                print("------------------------------------")
                print("Image iteration: " + str(image_iteration))
                print("Setting up opacity of all stimulis to 100%...")
                print("------------------------------------")
                for estimulo in estimulos:
                    estimulo.opacity = 1.0
                image_iteration += 1
                    
            if image_iteration < 8: # Todos los filtros
                estimulo_seleccionado = estimulos[image_iteration % len(estimulos)]
                estimulo_seleccionado.opacity = 0.0
                print("------------------------------------")
                print("Image iteration: " + str(image_iteration))
                print("Removing dots "+ str(image_iteration) + ".")
                print("------------------------------------")
                image_iteration += 1
                
            elif image_iteration == 8: # Ultimo filtro
                print("------------------------------------")
                print("Image iteration: " + str(image_iteration))
                print("Showing NO dots")
                print("------------------------------------")
            
            # keep track of which components have finished
            trial_1Components = [image, dots_1, dots_2, dots_3, dots_4, dots_5, dots_6, dots_7, key_resp_8]
            for thisComponent in trial_1Components:
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
            
            # --- Run Routine "trial_1" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image* updates
                
                # if image is starting this frame...
                if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image.frameNStart = frameN  # exact frame index
                    image.tStart = t  # local t and not account for scr refresh
                    image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image.started')
                    # update status
                    image.status = STARTED
                    image.setAutoDraw(True)
                
                # if image is active this frame...
                if image.status == STARTED:
                    # update params
                    pass
                
                # if image is stopping this frame...
                if image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image.tStartRefresh + time-frameTolerance:
                        # keep track of stop time/frame for later
                        image.tStop = t  # not accounting for scr refresh
                        image.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image.stopped')
                        # update status
                        image.status = FINISHED
                        image.setAutoDraw(False)
                
                # *dots_1* updates
                
                # if dots_1 is starting this frame...
                if dots_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    dots_1.frameNStart = frameN  # exact frame index
                    dots_1.tStart = t  # local t and not account for scr refresh
                    dots_1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(dots_1, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    dots_1.status = STARTED
                    dots_1.setAutoDraw(True)
                
                # if dots_1 is active this frame...
                if dots_1.status == STARTED:
                    # update params
                    pass
                
                # if dots_1 is stopping this frame...
                if dots_1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > dots_1.tStartRefresh + time-frameTolerance:
                        # keep track of stop time/frame for later
                        dots_1.tStop = t  # not accounting for scr refresh
                        dots_1.frameNStop = frameN  # exact frame index
                        # update status
                        dots_1.status = FINISHED
                        dots_1.setAutoDraw(False)
                
                # *dots_2* updates
                
                # if dots_2 is starting this frame...
                if dots_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    dots_2.frameNStart = frameN  # exact frame index
                    dots_2.tStart = t  # local t and not account for scr refresh
                    dots_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(dots_2, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    dots_2.status = STARTED
                    dots_2.setAutoDraw(True)
                
                # if dots_2 is active this frame...
                if dots_2.status == STARTED:
                    # update params
                    pass
                
                # if dots_2 is stopping this frame...
                if dots_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > dots_2.tStartRefresh + time-frameTolerance:
                        # keep track of stop time/frame for later
                        dots_2.tStop = t  # not accounting for scr refresh
                        dots_2.frameNStop = frameN  # exact frame index
                        # update status
                        dots_2.status = FINISHED
                        dots_2.setAutoDraw(False)
                
                # *dots_3* updates
                
                # if dots_3 is starting this frame...
                if dots_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    dots_3.frameNStart = frameN  # exact frame index
                    dots_3.tStart = t  # local t and not account for scr refresh
                    dots_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(dots_3, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    dots_3.status = STARTED
                    dots_3.setAutoDraw(True)
                
                # if dots_3 is active this frame...
                if dots_3.status == STARTED:
                    # update params
                    pass
                
                # if dots_3 is stopping this frame...
                if dots_3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > dots_3.tStartRefresh + time-frameTolerance:
                        # keep track of stop time/frame for later
                        dots_3.tStop = t  # not accounting for scr refresh
                        dots_3.frameNStop = frameN  # exact frame index
                        # update status
                        dots_3.status = FINISHED
                        dots_3.setAutoDraw(False)
                
                # *dots_4* updates
                
                # if dots_4 is starting this frame...
                if dots_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    dots_4.frameNStart = frameN  # exact frame index
                    dots_4.tStart = t  # local t and not account for scr refresh
                    dots_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(dots_4, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    dots_4.status = STARTED
                    dots_4.setAutoDraw(True)
                
                # if dots_4 is active this frame...
                if dots_4.status == STARTED:
                    # update params
                    pass
                
                # if dots_4 is stopping this frame...
                if dots_4.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > dots_4.tStartRefresh + time-frameTolerance:
                        # keep track of stop time/frame for later
                        dots_4.tStop = t  # not accounting for scr refresh
                        dots_4.frameNStop = frameN  # exact frame index
                        # update status
                        dots_4.status = FINISHED
                        dots_4.setAutoDraw(False)
                
                # *dots_5* updates
                
                # if dots_5 is starting this frame...
                if dots_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    dots_5.frameNStart = frameN  # exact frame index
                    dots_5.tStart = t  # local t and not account for scr refresh
                    dots_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(dots_5, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    dots_5.status = STARTED
                    dots_5.setAutoDraw(True)
                
                # if dots_5 is active this frame...
                if dots_5.status == STARTED:
                    # update params
                    pass
                
                # if dots_5 is stopping this frame...
                if dots_5.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > dots_5.tStartRefresh + time-frameTolerance:
                        # keep track of stop time/frame for later
                        dots_5.tStop = t  # not accounting for scr refresh
                        dots_5.frameNStop = frameN  # exact frame index
                        # update status
                        dots_5.status = FINISHED
                        dots_5.setAutoDraw(False)
                
                # *dots_6* updates
                
                # if dots_6 is starting this frame...
                if dots_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    dots_6.frameNStart = frameN  # exact frame index
                    dots_6.tStart = t  # local t and not account for scr refresh
                    dots_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(dots_6, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    dots_6.status = STARTED
                    dots_6.setAutoDraw(True)
                
                # if dots_6 is active this frame...
                if dots_6.status == STARTED:
                    # update params
                    pass
                
                # if dots_6 is stopping this frame...
                if dots_6.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > dots_6.tStartRefresh + time-frameTolerance:
                        # keep track of stop time/frame for later
                        dots_6.tStop = t  # not accounting for scr refresh
                        dots_6.frameNStop = frameN  # exact frame index
                        # update status
                        dots_6.status = FINISHED
                        dots_6.setAutoDraw(False)
                
                # *dots_7* updates
                
                # if dots_7 is starting this frame...
                if dots_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    dots_7.frameNStart = frameN  # exact frame index
                    dots_7.tStart = t  # local t and not account for scr refresh
                    dots_7.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(dots_7, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    dots_7.status = STARTED
                    dots_7.setAutoDraw(True)
                
                # if dots_7 is active this frame...
                if dots_7.status == STARTED:
                    # update params
                    pass
                
                # if dots_7 is stopping this frame...
                if dots_7.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > dots_7.tStartRefresh + time-frameTolerance:
                        # keep track of stop time/frame for later
                        dots_7.tStop = t  # not accounting for scr refresh
                        dots_7.frameNStop = frameN  # exact frame index
                        # update status
                        dots_7.status = FINISHED
                        dots_7.setAutoDraw(False)
                
                # *key_resp_8* updates
                waitOnFlip = False
                
                # if key_resp_8 is starting this frame...
                if key_resp_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_8.frameNStart = frameN  # exact frame index
                    key_resp_8.tStart = t  # local t and not account for scr refresh
                    key_resp_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_8, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_8.started')
                    # update status
                    key_resp_8.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_8.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_resp_8 is stopping this frame...
                if key_resp_8.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp_8.tStartRefresh + time-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp_8.tStop = t  # not accounting for scr refresh
                        key_resp_8.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp_8.stopped')
                        # update status
                        key_resp_8.status = FINISHED
                        key_resp_8.status = FINISHED
                if key_resp_8.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_8.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_8_allKeys.extend(theseKeys)
                    if len(_key_resp_8_allKeys):
                        key_resp_8.keys = _key_resp_8_allKeys[-1].name  # just the last key pressed
                        key_resp_8.rt = _key_resp_8_allKeys[-1].rt
                        key_resp_8.duration = _key_resp_8_allKeys[-1].duration
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
                for thisComponent in trial_1Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_1" ---
            for thisComponent in trial_1Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trial_1.stopped', globalClock.getTime())
            # check responses
            if key_resp_8.keys in ['', [], None]:  # No response was made
                key_resp_8.keys = None
            filter_removal.addData('key_resp_8.keys',key_resp_8.keys)
            if key_resp_8.keys != None:  # we had a response
                filter_removal.addData('key_resp_8.rt', key_resp_8.rt)
                filter_removal.addData('key_resp_8.duration', key_resp_8.duration)
            # the Routine "trial_1" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
        # completed 8.0 repeats of 'filter_removal'
        
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
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
