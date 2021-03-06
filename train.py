from pathlib import Path

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
	
	# If the action is different from the previous make changes otherwise do nothing
	releasekeys(curr, keyb)
	curr = action_dict[action]

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
	GAMMA = 0.9
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
		'walk': 0,
		'jump': 0,
		'fall': 0,
		'small to big': 5,
		'big to fire': 2,
		'big to small': -1,
		'flag pole': 10,
		'walking to castle': 10,
		'end of level fall': 10,
		'death jump': -1}
	REWARDS_A = {0: -1,
				1: 1,
				2: -1,
				3: 0,
				4: 1,
				5: -1}
	REWARDS_G = {
	'main menu': 0,
	'load screen': 0,
	'time out': -10,
	'game over': -5,
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
	
	dqn_net = cnn.Net().cuda()

	FILE_CHECK = Path(SAVE_PATH_CNN)
	if FILE_CHECK.exists():
		print('----- Found an existing CNN')
		trained_params = torch.load(SAVE_PATH_CNN)
		dqn_net.load_state_dict(trained_params)
	optimizer = optim.Adam(dqn_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
	nn_loss = torch.nn.MSELoss()

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
		EPISODE_SS = []
		EPISODE_STATES = []
		EPISODE_ACTIONS = []
		while GAME_STATE == 'level1' and MARIO_STATE != 'death jump':
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

			# Save episode information
			if EPISODE_ACTIONS:
				EPISODE_SS.append( state )
				EPISODE_ACTIONS.append( [a] )
				EPISODE_STATES.append( [GAME_STATE, MARIO_STATE] )
			else:
				EPISODE_SS = [state]
				EPISODE_ACTIONS = [ [a] ]
				EPISODE_STATES = [ [GAME_STATE, MARIO_STATE] ]				

			# time.sleep(0.25)
			tend = time.time()
			print( q_values )
			print('Processing Time: %.3f sec || Action: %i || MS: %s' %(tend - tstart, a, MARIO_STATE))

		# Release all keys
		CURRENT_ACTION = action2key(0, CURRENT_ACTION, KEYBOARD, ACTION_OPTIONS)

		############################################
		# Train on Episode + Experience Replay
		############################################
		# Note the game should train on 3 different episodes before it 
		print( '[MAIN] Training Model...' )
		N = len( EPISODE_ACTIONS )

		for iter in range(N-1):
			a = EPISODE_ACTIONS[iter]

			optimizer.zero_grad()

			state = EPISODE_SS[iter]
			state_tensor = torch.unsqueeze( torch.Tensor( state ), 0 )
			state_tensor = torch.unsqueeze( state_tensor, 0 )
			cnn_input = Variable( state_tensor.cuda() )
			q_val = dqn_net( cnn_input )

			state = EPISODE_SS[iter+1]
			state_tensor = torch.unsqueeze( torch.Tensor( state ), 0 )
			state_tensor = torch.unsqueeze( state_tensor, 0 )
			cnn_input = Variable( state_tensor.cuda() )
			q_valp = dqn_net( cnn_input ).detach()

			# Set up constants for the loss
			max_val, max_ind = torch.max(q_valp.data, 1)
			max_ind = max_ind.cpu().numpy()[0][0]

			q_val_copy = q_val.data.clone()
			q_val_copy[0, max_ind] = GAMMA * max_val[0][0]
			q_val_copy = Variable(q_val_copy, requires_grad=False)

			#Define the rewards
			gs = EPISODE_STATES[iter+1][0]
			ms = EPISODE_STATES[iter+1][1]
			r = torch.zeros(q_val.size())
			r[0, max_ind] = REWARDS_G[gs] + REWARDS_M[ms] + REWARDS_A[a[0]]
			r = Variable(r, requires_grad=False).cuda()

			q_val.data.cpu()
			loss = nn_loss(q_val, r + q_val_copy)
			loss.backward()
			optimizer.step()

		print('[MAIN] Done!')
		# Save the results
		torch.save(dqn_net.state_dict(), SAVE_PATH_CNN)
		print('[MAIN] Model Saved s: ', SAVE_PATH_CNN)

		# Wait until the game is back to the main menu to start again
		MARIO_STATE = 'standing'
		while GAME_STATE != 'main menu':
			# Get new GameState
			file = open(FILENAME_GAME, 'r')
			GAME_STATE = file.read()
			time.sleep(0.5)