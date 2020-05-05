
class Game():

	def __init__(self, game_name):
		self.env = gym.make(game_name)
		self.env.reset()
		self.n_actions = self.env.action_space.n
		_, _, self.scr_h, self.scr_w = self.get_screen().shape

	def get_screen():
		# Gym reutns screen as HWC, convert it to CHW
		screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
		_, scr_h, scr_w = screen.shape

		# Perform crops

		# Convert to float, rescale
		screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.

		# TODO(Harshan): Add PyTorch support

	def select_action(action):
		raise NotImplementedError
