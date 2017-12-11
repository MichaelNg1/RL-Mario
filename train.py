#Import keyboard dependencies
import pykeyboard
import pymouse
import time

#Import Screenshot dependencies
import pyscreeze
import time
from PIL import Image

##########################################################################################
#NOTE: the game needs to be running prior to this script's execution
##########################################################################################

############################################
# action2key: executes the action
############################################
def action2key(action, curr, keyb, action_dict):
	
	# Release keys from previous commands
	if curr:
		for el in curr:
			keyb.release_key( el )

	# Assign commands based on "action" where action is an int (4 total)
	curr = action_dict[action]

	# Execute the actions
	if curr:
		for el in curr:
			keyb.press_key( el )

	return curr

def screenshot(loc, filename):
	x = loc[0]
	y = loc[1]
	width = loc[2]
	height = loc[3]
	height_compensation = 0
	y -= height_compensation
	height += height_compensation
	location_region = (x, y, width, height)

	# Test screenshot
	im = pyscreeze.screenshot(region=location_region)
	im_resize = im.resize((84, 84), Image.ANTIALIAS)
	im_gray = im_resize.convert('L')
	im_gray.save( filename )

############################################
# main: executes the game
############################################
if __name__ == "__main__":

	#Define constants
	KEYBOARD = pykeyboard.PyKeyboard()
	MOUSE = pymouse.PyMouse()
	CURRENT_ACTION = []
	ACTION_OPTIONS = {0: [],
				1: [KEYBOARD.right_key],
				2: [KEYBOARD.left_key],
				3: ['a'],
				4: [KEYBOARD.right_key, 'a'],
				5: [KEYBOARD.left_key, 'a']}

	print( '[MAIN] Locating Game Screen' )
	# Finds the game screen to take screenshots
	while True:
		location = pyscreeze.locateOnScreen('locate_window.png')
		print (location)
		if location:
			print( '[MAIN] Screenshot success' )
			break

	screenshot(location, 'gray.png')
	time.sleep(0.5)

	print( '[MAIN] Starting Game' )

	# Start the game if in menu and pause for one second
	MOUSE.click(location[0], location[1], 1)
	CURRENT_ACTION = action2key(3, CURRENT_ACTION, KEYBOARD, ACTION_OPTIONS)
	time.sleep(0.5324)
	CURRENT_ACTION = action2key(0, CURRENT_ACTION, KEYBOARD, ACTION_OPTIONS)

	# Wait for the game to load
	while True:
		l= pyscreeze.locateOnScreen('START_FLAG.png')
		print (l)
		if l:
			print( '[MAIN] EPISODE START' )
			break


	

