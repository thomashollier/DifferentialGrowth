import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import matplotlib.animation as animation
import argparse
from  DifferentialGrowth import *
 


class DifferentialGrowth:
	def __init__(self, num_initial_points=50, radius=1.0, min_dist=0.1, max_dist=0.2,
				 base_repulsion_radius=0.3, repulsion_strength=0.1, attraction_strength=0.1,
				 bounds=(-2, 2), verbosity=1, noise_level=0.0, random_seed=None):
		self.verbosity = verbosity
		
		if random_seed is not None:
			np.random.seed(random_seed)
			
		if self.verbosity >= 1:
			print(f"Initializing with {num_initial_points} points on circle of radius {radius}")
		
		# Initialize points in a circle
		angles = np.linspace(0, 2*np.pi, num_initial_points)
		x = radius * np.cos(angles)
		y = radius * np.sin(angles)
		self.points = np.column_stack((x, y))
		
		# Add initial noise to points
		if noise_level > 0:
			noise = np.random.normal(0, noise_level * radius, self.points.shape)
			self.points += noise
		
		# Initialize per-particle repulsion radii with noise
		self.base_repulsion_radius = base_repulsion_radius
		self.particle_radii = np.ones(num_initial_points) * base_repulsion_radius
		if noise_level > 0:
			radius_noise = np.random.normal(0, noise_level * base_repulsion_radius, num_initial_points)
			self.particle_radii += radius_noise
			self.particle_radii = np.clip(self.particle_radii, base_repulsion_radius * 0.5, base_repulsion_radius * 2.0)
		
		# Parameters
		self.min_dist = min_dist
		self.max_dist = max_dist
		self.repulsion_strength = repulsion_strength
		self.attraction_strength = attraction_strength
		self.bounds = bounds
		self.noise_level = noise_level
		
		if self.verbosity >= 1:
			print("Parameters:")
			print(f"  min_dist: {self.min_dist}")
			print(f"  max_dist: {self.max_dist}")
			print(f"  base_repulsion_radius: {self.base_repulsion_radius}")
			print(f"  repulsion_strength: {self.repulsion_strength}")
			print(f"  attraction_strength: {self.attraction_strength}")
			print(f"  bounds: {self.bounds}")
			print(f"  noise_level: {self.noise_level}")
			print(f"  particle radius range: [{np.min(self.particle_radii):.3f}, {np.max(self.particle_radii):.3f}]")

	def calculate_repulsion(self, point_idx):
		point = self.points[point_idx]
		diffs = self.points - point
		distances = np.linalg.norm(diffs, axis=1)
		
		# Use this particle's specific repulsion radius
		current_radius = self.particle_radii[point_idx]
		mask = (distances < current_radius) & (distances > 0)
		
		if not np.any(mask):
			if self.verbosity >= 2:
				print(f"Point {point_idx} has no nearby points for repulsion")
			return np.zeros(2)
			
		valid_diffs = diffs[mask]
		valid_distances = distances[mask]
		
		num_affecting = np.sum(mask)
		if self.verbosity >= 2:
			print(f"Point {point_idx} being repelled by {num_affecting} nearby points")
		
		normalized_diffs = valid_diffs / valid_distances[:, np.newaxis]
		strength = (1 - valid_distances / current_radius)[:, np.newaxis]
		
		# Add noise to directions
		if self.noise_level > 0:
			noise = np.random.normal(0, self.noise_level, normalized_diffs.shape)
			normalized_diffs += noise
		
		total_repulsion = np.sum(normalized_diffs * strength, axis=0)
		
		if self.verbosity >= 2:
			print(f"  Total repulsion force: {total_repulsion}")
		return total_repulsion * self.repulsion_strength

	def line_segments_intersect(self, p1, p2, p3, p4):
		"""Check if line segments (p1,p2) and (p3,p4) intersect"""
		def ccw(A, B, C):
			return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
		return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

	def fix_self_intersections(self):
		"""Fix self-intersections by pushing intersecting segments apart"""
		num_points = len(self.points)
		moved = False
		
		# Check each pair of non-adjacent segments
		for i in range(num_points):
			p1 = self.points[i]
			p2 = self.points[(i + 1) % num_points]
			
			# Start checking from 2 segments ahead to avoid adjacent segments
			for j in range(i + 2, num_points):
				# Skip if this would check against the starting segment or its neighbor
				if (j + 1) % num_points == i:
					continue
					
				p3 = self.points[j]
				p4 = self.points[(j + 1) % num_points]
				
				if self.line_segments_intersect(p1, p2, p3, p4):
					# Calculate midpoints
					mid1 = (p1 + p2) / 2
					mid2 = (p3 + p4) / 2
					
					# Calculate push direction
					direction = mid2 - mid1
					distance = np.linalg.norm(direction)
					if distance < 1e-6:
						# If points are too close, push in a random direction
						angle = np.random.uniform(0, 2*np.pi)
						direction = np.array([np.cos(angle), np.sin(angle)])
					else:
						direction /= distance
					
					# Push segments apart
					push_strength = self.min_dist
					self.points[i] -= direction * push_strength * 0.5
					self.points[(i + 1) % num_points] -= direction * push_strength * 0.5
					self.points[j] += direction * push_strength * 0.5
					self.points[(j + 1) % num_points] += direction * push_strength * 0.5
					
					moved = True
		
		return moved

	def save_svg(self, filename="growth.svg", width=800, height=800):
		"""Save current state as SVG with smooth spline curves"""
		# Calculate view boundaries with padding
		padding = 0.2  # Increased padding for better framing
		x_min, y_min = np.min(self.points, axis=0) - padding
		x_max, y_max = np.max(self.points, axis=0) + padding
		
		# Create viewBox with proper aspect ratio
		view_width = x_max - x_min
		view_height = y_max - y_min
		viewBox = f"{x_min} {y_min} {view_width} {view_height}"
		
		# Create SVG header with refined styling
		svg = [
			f'<?xml version="1.0" encoding="UTF-8"?>',
			f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="{viewBox}">',
			'<g fill="black" stroke="red" stroke-width="0.02" stroke-linejoin="round">'
		]
		
		# Create points list for the path, including wrapping points for smooth closure
		points = np.vstack((
			self.points[-1],  # Add last point at start
			self.points,	  # Original points
			self.points[0],   # Add first point at end
			self.points[1]	# Add second point at end
		))
		
		# Generate path data using cubic bezier curves
		path_data = ["M"]
		
		# Start at the first actual point
		path_data.extend([f"{points[1][0]},{points[1][1]}"])
		
		# Helper function to calculate smooth tangent
		def calculate_tangent(p0, p1, p2):
			"""Calculate a smooth tangent vector at p1 based on p0 and p2"""
			v1 = p1 - p0
			v2 = p2 - p1
			# Average the vectors and scale by tension
			tension = 0.25  # Adjust this value to control curve tightness (0.0-1.0)
			tangent = (v1 + v2) * tension
			return tangent
		
		# Generate smooth bezier curves
		for i in range(1, len(points) - 2):
			# Get points and calculate tangents
			p0 = points[i-1]
			p1 = points[i]
			p2 = points[i+1]
			p3 = points[i+2]
			
			# Calculate tangents using surrounding points
			tangent1 = calculate_tangent(p0, p1, p2)
			tangent2 = calculate_tangent(p1, p2, p3)
			
			# Calculate control points
			cp1 = p1 + tangent1
			cp2 = p2 - tangent2
			
			# Add cubic bezier curve command
			path_data.extend([
				"C",  # Cubic bezier
				f"{cp1[0]},{cp1[1]}",	  # First control point
				f"{cp2[0]},{cp2[1]}",	  # Second control point
				f"{p2[0]},{p2[1]}"		 # End point
			])
		
		# Close the path
		path_data.append("Z")
		
		# Add path to SVG
		svg.append(f'<path d="{" ".join(path_data)}"/>')
		
		# Close SVG
		svg.extend(['</g>', '</svg>'])
		
		# Write to file
		with open(filename, 'w') as f:
			f.write('\n'.join(svg))
			
		if self.verbosity >= 1:
			print(f"Saved SVG to {filename}")

	def grow(self):
		if self.verbosity >= 1:
			print("\nStarting growth iteration")
			print(f"Current number of points: {len(self.points)}")
		
		# Split edges that are too long
		if len(self.points) > 1500:
			print("more than 1500 points, quitting")
			return
		split_indices = []
		split_radii = []
		for i in range(len(self.points)):
			next_i = (i + 1) % len(self.points)
			dist = np.linalg.norm(self.points[i] - self.points[next_i])
			if dist > self.max_dist:
				if self.noise_level > 0:
					noise = np.random.normal(0, self.noise_level, 2)
					midpoint = (self.points[i] + self.points[next_i]) / 2 + noise
				else:
					midpoint = (self.points[i] + self.points[next_i]) / 2
				
				inherited_radius = (self.particle_radii[i] + self.particle_radii[next_i]) / 2
				split_indices.append((i, midpoint))
				split_radii.append(inherited_radius)
		
		if self.verbosity >= 1:
			print(f"Number of edges to split: {len(split_indices)}")
		
		for idx, (split_idx, midpoint) in enumerate(reversed(split_indices)):
			self.points = np.insert(self.points, split_idx + 1, midpoint, axis=0)
			self.particle_radii = np.insert(self.particle_radii, split_idx + 1, split_radii[-idx-1])
		
		if self.verbosity >= 1:
			print(f"New number of points after splitting: {len(self.points)}")
			print(f"Particle radius range: [{np.min(self.particle_radii):.3f}, {np.max(self.particle_radii):.3f}]")
		
		forces = np.zeros_like(self.points)
		
		if self.verbosity >= 2:
			print("\nCalculating repulsion forces:")
		for i in range(len(self.points)):
			repulsion_force = self.calculate_repulsion(i)
			forces[i] -= repulsion_force
		
		if self.verbosity >= 2:
			print("\nCalculating attraction forces:")
		
		for i in range(len(self.points)):
			prev_i = (i - 1) % len(self.points)
			next_i = (i + 1) % len(self.points)
			
			to_prev = self.points[prev_i] - self.points[i]
			to_next = self.points[next_i] - self.points[i]
			
			if self.noise_level > 0:
				noise_prev = np.random.normal(0, self.noise_level, 2)
				noise_next = np.random.normal(0, self.noise_level, 2)
				to_prev += noise_prev
				to_next += noise_next
			
			attraction_force = (to_prev + to_next) * self.attraction_strength
			forces[i] += attraction_force
			
			if self.verbosity >= 2:
				print(f"Point {i} attraction force: {attraction_force}")
		
		if self.verbosity >= 1:
			print("\nUpdating positions:")
			max_force = np.max(np.abs(forces))
			print(f"Maximum force magnitude: {max_force}")
		
		# Apply forces
		self.points += forces
		
		# Fix any self-intersections
		max_intersection_fixes = 5
		fix_count = 0
		while self.fix_self_intersections() and fix_count < max_intersection_fixes:
			fix_count += 1
			if self.verbosity >= 1:
				print(f"Fixed self-intersections (attempt {fix_count})")
		
		# Keep points within bounds
		original_points = self.points.copy()
		self.points = np.clip(self.points, self.bounds[0], self.bounds[1])
		num_constrained = np.sum(self.points != original_points)
		if num_constrained > 0 and self.verbosity >= 1:
			print(f"Constrained {num_constrained} points back within bounds")

	def plot(self):
		plt.style.use('dark_background')
		fig = plt.figure(figsize=(8, 8))
		ax = fig.add_subplot(111)

		# Create closed polygon for filling
		x = np.append(self.points[:, 0], self.points[0, 0])
		y = np.append(self.points[:, 1], self.points[0, 1])

		# Fill the shape
		ax.fill(x, y, color='white', alpha=0.2)

		# Draw the outline
		ax.plot(x, y, 'w-', linewidth=1)

		ax.set_facecolor('black')
		fig.patch.set_facecolor('black')
		plt.axis('equal')
		plt.grid(True, color='gray', alpha=0.2)
		plt.show()
