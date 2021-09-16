'''
This scipt deletes the local models before initializing a new training session
'''

import os
 
dir = 'model'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))