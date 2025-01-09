# Save this as differential_growth.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import matplotlib.animation as animation
import argparse
from  DifferentialGrowth import *



#[Previous DifferentialGrowth class implementation goes here]

def animate_growth(frames=200, verbosity=1, **kwargs):
	if verbosity >= 1:
		print(f"\nStarting animation with {frames} frames")
			
	system = DifferentialGrowth(verbosity=verbosity, **kwargs)
	
	plt.style.use('dark_background')
	fig, ax = plt.subplots(figsize=(8, 8))
	ax.set_xlim(system.bounds)
	ax.set_ylim(system.bounds)
	ax.set_facecolor('black')
	fig.patch.set_facecolor('black')
	ax.grid(True, color='gray', alpha=0.2)
	
	# Create two artists: one for fill and one for line
	fill = ax.fill([], [], color='white', alpha=0.2)[0]
	line, = ax.plot([], [], 'w-', linewidth=1)
	
	def init():
		line.set_data([], [])
		fill.set_xy(np.array([[0, 0]]))
		return line, fill
	
	class AnimationHandler:
		def __init__(self):
			self.frame = 0
			
		def update(self, frame):
			if verbosity >= 1:
				print(f"\nAnimating frame {self.frame}")
			system.grow()
			
			# Save SVG if requested for this frame
			if args.save_svg and (self.frame == args.save_frame or 
								(args.save_frame == -1 and frame == frames - 1)):
				if verbosity >= 1:
					print(f"Saving frame {self.frame} as SVG: {args.save_svg}")
				system.save_svg(args.save_svg)
			
			self.frame += 1
			x = np.append(system.points[:, 0], system.points[0, 0])
			y = np.append(system.points[:, 1], system.points[0, 1])
			line.set_data(x, y)
			fill.set_xy(np.column_stack((x, y)))
			return line, fill
	
	handler = AnimationHandler()
	anim = animation.FuncAnimation(fig, handler.update, frames=frames,
								 init_func=init, blit=True,
								 interval=50)
	plt.show()
	return anim

def parse_args():
	parser = argparse.ArgumentParser(description='2D Differential Growth Simulation')
	parser.add_argument('-v', '--verbosity', type=int, choices=[0, 1, 2], default=1,
					  help='Verbosity level: 0=minimal, 1=normal, 2=detailed (default: 1)')
	
	parser.add_argument('--num-points', type=int, default=115,
						help='Number of initial points (default: 115)')
	parser.add_argument('--radius', type=float, default=0.5,
						help='Initial circle radius (default: 0.5)')
	parser.add_argument('--min-dist', type=float, default=0.02,
						help='Minimum distance between points (default: 0.02)')
	parser.add_argument('--max-dist', type=float, default=0.2,
						help='Maximum distance between points (default: 0.2)')
	parser.add_argument('--repulsion-radius', type=float, default=0.8,
						help='Base radius for point repulsion (default: 0.8)')
	parser.add_argument('--repulsion-strength', type=float, default=0.025,
						help='Strength of repulsion force (default: 0.025)')
	parser.add_argument('--attraction-strength', type=float, default=0.025,
						help='Strength of attraction force (default: 0.025)')
	parser.add_argument('--bounds', type=float, default=20.0,
						help='Boundary limits (default: 20.0)')
	parser.add_argument('--iterations', type=int, default=30,
						help='Number of growth iterations (default: 30)')
	parser.add_argument('--animate', action='store_true',
						help='Create animation instead of static plot')
	parser.add_argument('--frames', type=int, default=35,
						help='Number of animation frames (default: 25)')
	parser.add_argument('--noise', type=float, default=0.0021,
						help='Level of randomness in the system (default: 0.0021)')
	parser.add_argument('--seed', type=int, default=666,
						help='Random seed for reproducibility')
	parser.add_argument('--save-svg', type=str, default=None,
						help='Save specified frame as SVG file')
	parser.add_argument('--save-frame', type=int, default=-1,
						help='Frame number to save as SVG (-1 for final frame)')
	
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()
	
	if args.verbosity >= 1:
		print("Starting differential growth simulation")
	
	params = {
		'num_initial_points': args.num_points,
		'radius': args.radius,
		'min_dist': args.min_dist,
		'max_dist': args.max_dist,
		'base_repulsion_radius': args.repulsion_radius,
		'repulsion_strength': args.repulsion_strength,
		'attraction_strength': args.attraction_strength,
		'bounds': (-args.bounds, args.bounds),
		'verbosity': args.verbosity,
		'noise_level': args.noise,
		'random_seed': args.seed
	}
	

	mnds = [.02, .01, .015, .025, .03]
	mxds = [.2, .15, .175, .225, .25,.3]
	rads = [.8, 1, .9, .3, .4, .5, .6, .7]

	mnds = [.15]
	mxds = [.2]
	rads = [.8]

	for mnd in mnds:
		params['min_dist'] = mnd
		for mxd in mxds: 
			params['max_dist'] = mxd
			for r in rads:
				params['base_repulsion_radius'] = r
				system = DifferentialGrowth(**params)
				if args.verbosity >= 1:
					print(f"\nPerforming {args.iterations} growth iterations")
				for iteration in range(args.iterations):
					if args.verbosity >= 1:
						print(f"\nIteration {iteration + 1}/{args.iterations}")
					system.grow()
					
					# Save SVG if requested for this iteration
					args.save_svg = True
					if args.save_svg and (iteration == args.save_frame or 
											(args.save_frame == -1 and iteration == args.iterations - 1)):
						svgFile = "thing__%s_%s_%s.svg" % (str(mnd).lstrip(), str(mxd).lstrip(), str(r).lstrip())
						if args.verbosity >= 1:
							print(f"Saving iteration {iteration} as SVG: {svgFile}")
						system.save_svg(svgFile)
							
