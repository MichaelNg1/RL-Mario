#Import keyboard dependencies
import pykeyboard
import pymouse
import time

#Import Screenshot dependencies
import pyscreeze
import time
from PIL import Image

#Import Pytorch
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#Import local scripts
import CNN as cnn

##########################################################################################
#NOTE: the game needs to be running prior to this script's execution
##########################################################################################

##########################################################################################
# action2key: executes the action
##########################################################################################
def action2key(action, curr, keyb, action_dict):
	
	# Release keys from previous commands
	releasekeys(curr, keyb)

	# Assign commands based on "action" where action is an int (4 total)
	curr = action_dict[action]

	# Execute the actions
	if curr:
		for el in curr:
			keyb.press_key( el )

	return curr

def releasekeys(curr, keyb):
	if curr:
		for el in curr:
			keyb.release_key( el )
##########################################################################################
# screenshot: returns a processed image and saves its according to filename
##########################################################################################
def screenshot(loc):
	x = loc[0]
	y = loc[1]
	width = loc[2]
	height = loc[3]
	# height_compensation = 0
	# y -= height_compensation
	# height += height_compensation
	location_region = (x, y, width, height)

	# Test screenshot
	im = pyscreeze.screenshot(region=location_region)
	# im_resize = im.resize((84, 84), Image.ANTIALIAS)
	im_resize = im.resize((84, 84))
	im_gray = im_resize.convert('L')
	
	return im_gray

##########################################################################################
# main: executes the game
##########################################################################################
if __name__ == "__main__":

	############################################
	#Define constants
	############################################
	EPOCH = 100
	KEYBOARD = pykeyboard.PyKeyboard()
	MOUSE = pymouse.PyMouse()
	CURRENT_ACTION = []
	ACTION_OPTIONS = {0: [],
				1: [KEYBOARD.right_key],
				2: [KEYBOARD.left_key],
				3: ['a'],
				4: [KEYBOARD.right_key, 'a'],
				5: [KEYBOARD.left_key, 'a']}
	NUM_ACTIONS = 6
	EPS = 0.1
	REWARDS_M = {
		'standing': 0,
		'walk': 1,
		'jump': 1,
		'fall': 0,
		'small to big': 2,
		'big to fire': 2,
		'big to small': -1,
		'flag pole': 10,
		'walking to castle': 10,
		'end of level fall': 10,
		'death jump': -10}
	REWARDS_G = {
	'main menu': 0,
	'load screen': 0,
	'time out': -100,
	'game over': -20,
	'level1': 0}

	SAVE_PATH_CNN = 'CNN.tar'

	# The possible gamestates are:
	# MAIN_MENU = 'main menu'
	# LOAD_SCREEN = 'load screen'
	# TIME_OUT = 'time out'
	# GAME_OVER = 'game over'
	# LEVEL1 = 'level1'
	GAME_STATE = 'main menu'
	FILENAME_GAME = 'game_state.txt'

	# The possible mario states are:
	# STAND = 'standing'
	# WALK = 'walk'
	# JUMP = 'jump'
	# FALL = 'fall'
	# SMALL_TO_BIG = 'small to big'
	# BIG_TO_FIRE = 'big to fire'
	# BIG_TO_SMALL = 'big to small'
	# FLAGPOLE = 'flag pole'
	# WALKING_TO_CASTLE = 'walking to castle'
	# END_OF_LEVEL_FALL = 'end of level fall'
	MARIO_STATE = 'standing'
	FILENAME_MARIO = 'mario_state.txt'

	############################################
	# Set up the CNN
	############################################
	dqn_net = cnn.Net()
	dqn_net = dqn_net.cuda()
	optimizer = optim.Adam(dqn_net.parameters(), lr=0.0002, betas=(0.5, 0.999))

	############################################
	# Find the Game Screen
	############################################
	print( '[MAIN] Locating Game Screen' )
	# Finds the game screen to take screenshots
	while True:
		location = pyscreeze.locateOnScreen('locate_window.png')
		print (location)
		if location:
			print( '[MAIN] Locating Game Screen: Success!' )
			break

	############################################
	# Train the CNN
	############################################
	for iter in range(EPOCH):
		# Click on the window
		MOUSE.click(location[0], location[1], 1)

		# Start the game if in menu and pause for two seconds
		print( '[MAIN] Starting Game, (EPOCH: %i)' % (iter) )
		if GAME_STATE == 'main menu':
			
			CURRENT_ACTION = action2key(3, CURRENT_ACTION, KEYBOARD, ACTION_OPTIONS)
			time.sleep(0.5)
			CURRENT_ACTION = action2key(0, CURRENT_ACTION, KEYBOARD, ACTION_OPTIONS)
			time.sleep(2)

		# Wait until it is in the level
		while GAME_STATE != 'level1':
			# Get new GameState
			file = open(FILENAME_GAME, 'r')
			GAME_STATE = file.read()
			time.sleep(0.5)

		############################################
		# Collect data on episode
		############################################
		while GAME_STATE == 'level1':
			tstart = time.time()

			# Get Screenshot
			state = np.array( screenshot(location), dtype=float )
			state_tensor = torch.unsqueeze( torch.Tensor( state ), 0 )
			state_tensor = torch.unsqueeze( state_tensor, 0 )
			cnn_input = Variable( state_tensor.cuda() )

			q_values = dqn_net( cnn_input )
			q_values = q_values.data.cpu().numpy()
			a_max = np.argmax( q_values )
			p_eps = (EPS / NUM_ACTIONS) * np.ones( NUM_ACTIONS )
			p_eps[a_max] = (EPS / NUM_ACTIONS) + 1 - EPS
			a = int( np.random.choice( NUM_ACTIONS, 1, p=p_eps) )

			CURRENT_ACTION = action2key(a, CURRENT_ACTION, KEYBOARD, ACTION_OPTIONS)

			# Get new GameState
			file = open(FILENAME_GAME, 'r')
			content = file.read()
			if content != '':
				GAME_STATE = content

			# Get new MarioState
			file = open(FILENAME_MARIO, 'r')
			content = file.read()
			if content != '':
				MARIO_STATE = content

			r = REWARDS_G[GAME_STATE] + REWARDS_M[MARIO_STATE]

			# time.sleep(0.25)
			tend = time.time()
			print('Processing Time: %.3f sec || Action: %i' %(tend - tstart, a))

		# Release all keys
		CURRENT_ACTION = action2key(0, CURRENT_ACTION, KEYBOARD, ACTION_OPTIONS)

		############################################
		# Train on Episode + Experience Replay
		############################################
		# Note the game should train on 3 different episodes before it 
		print( '[MAIN] Training Model...' )

		
		# Wait until the game is back to the main menu to start again
		while GAME_STATE != 'main menu':
			# Get new GameState
			file = open(FILENAME_GAME, 'r')
			GAME_STATE = file.read()
			time.sleep(0.5)