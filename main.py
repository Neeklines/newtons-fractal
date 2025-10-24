from tkinter.colorchooser import askcolor
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
import time

class NewtonFractalInteractive:
    def __init__(self):
        # Configuration
        self.n = 3
        self.xmin, self.xmax = -1, 1
        self.ymin, self.ymax = -1, 1
        self.res = 1000
        self.tol = 1e-6
        self.max_iter = 1000
        self.mode = "both"  # "root", "iter", "both"
        self.iter_color_mode = "hist"  # old, hsv, hist

        # Polynomial and roots
        self.f = np.poly1d([1] + [0] * (self.n - 1) + [-1])
        self.df = np.polyder(self.f)
        self.roots = self.f.r

        # Dynamic color map (root -> RGB color tuple)
        self.color_map = {}
        self.colorbar = None
        self.real_max_iter = None
        self.norm_iters = None

        # Colormap and brightness per mode
        self.mode_colormaps = {
            "root": "hsv",
            "iter": "twilight",
            "both": "twilight"
        }
        self.mode_brightness = {
            "root": 0.7,
            "iter": 1.0,
            "both": 1.0
        }

        self.benchmark()
        self.benchmark("loop")

        # Tkinter root for color chooser
        self.tk_root = tk.Tk()
        self.tk_root.withdraw()

        # Matplotlib figure setup
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.legend = None
        self.root_markers = []
        self.legend_patches = []
        self.fractal_img = None

        self.initial_coords = [self.xmin, self.xmax, self.ymin, self.ymax]

        self.fig.tight_layout()
        self._plot()
        self.ax.set_position([0.15, 0.15, 0.7, 0.7])
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('button_press_event', self._on_colorbar_click)

        plt.show()

    def _generate_meshgrid(self):
        x = np.linspace(self.xmin, self.xmax, self.res)
        y = np.linspace(self.ymin, self.ymax, self.res) * 1j
        X, Y = np.meshgrid(x, y)
        return (X + Y).flatten(), X.shape

    def _compute_both_vectorized(self, arr):
        iterations = np.zeros_like(arr, dtype=int)
        root_ids = np.full_like(arr, -1, dtype=int)
        unconverged = np.ones_like(arr, dtype=bool)

        for i in range(self.max_iter):
            indices = np.flatnonzero(unconverged)
            if not len(indices):
                break

            z = arr[indices]
            distances = np.abs(z[:, None] - self.roots[None, :])
            closest = np.argmin(distances, axis=1)
            converged = np.min(distances, axis=1) < self.tol
            converged_indices = indices[converged]

            root_ids[converged_indices] = closest[converged] + 1
            iterations[converged_indices] = i
            unconverged[converged_indices] = False

            arr[indices] -= self.f(z) / self.df(z)

        return root_ids, iterations

    def _compute_both_loop(self, arr):
        iterations = np.zeros_like(arr, dtype=int)
        root_ids = np.full_like(arr, -1, dtype=int)
        unconverged = np.ones_like(arr, dtype=bool)

        for i in range(self.max_iter):
            for root_id, root in enumerate(self.roots, 1):
                close = np.abs(arr[unconverged] - root) < self.tol
                global_indices = np.flatnonzero(unconverged)[close]
                root_ids[global_indices] = root_id
                iterations[global_indices] = i
                unconverged[global_indices] = False

            if not np.any(unconverged):
                break

            arr[unconverged] -= self.f(arr[unconverged]) / self.df(arr[unconverged])

        return root_ids, iterations
    
    # Benchmark methods
    def benchmark(self, method="vectorized"):
        arr, _ = self._generate_meshgrid()
        if method == "vectorized":
            compute = self._compute_both_vectorized
        elif method == "loop":
            compute = self._compute_both_loop
        else:
            raise ValueError("method must be 'vectorized' or 'loop'")

        start = time.time()
        root_ids, iterations = compute(arr.copy())
        end = time.time()

        print(f"{method} method took {end - start:.3f} seconds")
        return root_ids, iterations

    def generate_fractal(self):
        arr, shape = self._generate_meshgrid()

        root_ids, iteration_counts = self._compute_both_vectorized(arr)
        self.last_root_ids = root_ids
        self.norm_root_ids = root_ids / len(self.roots)
        self.norm_iters = iteration_counts / iteration_counts.max()
        self.real_max_iter = iteration_counts.max()
        brightness = np.exp(-self.norm_iters * 5)
        self.last_brightness = brightness

        # If using root-based colormap (dict-style), initialize if empty
        if isinstance(self.color_map, dict) and not self.color_map:
            cmap = plt.get_cmap(self.mode_colormaps.get(self.mode, "twilight"))
            for i, root in enumerate(self.roots, 1):
                self.color_map[root] = cmap(i / len(self.roots))[:3]

        img = np.zeros((arr.shape[0], 3))

        for i, root in enumerate(self.roots, 1):
            mask = root_ids == i

            if self.mode == "root":
                if isinstance(self.color_map, dict):
                    base_color = self.color_map.get(root, (1, 1, 1))
                else:
                    # fallback: map i to a color from colormap
                    base_color = self.color_map(i / len(self.roots))[:3]
                img[mask] = np.array(base_color) * self.mode_brightness["root"]

            elif self.mode == "iter":
                cmap = self.color_map if not isinstance(self.color_map, dict) else plt.get_cmap(self.mode_colormaps["iter"])
                img[mask] = cmap(self.norm_iters[mask])[:, :3] * self.mode_brightness["iter"]

            elif self.mode == "both":
                if isinstance(self.color_map, dict):
                    base_color = self.color_map.get(root, (1, 1, 1))
                else:
                    base_color = self.color_map(i / len(self.roots))[:3]
                img[mask] = base_color * (brightness[mask, np.newaxis] * self.mode_brightness["both"])

            else:
                if isinstance(self.color_map, dict):
                    base_color = self.color_map.get(root, (1, 1, 1))
                else:
                    base_color = self.color_map(i / len(self.roots))[:3]
                img[mask] = base_color

        return img.reshape(*shape, 3)

    
    def _plot(self):
        self.ax.clear()
        fractal = self.generate_fractal()
        extent = [self.xmin, self.xmax, self.ymin, self.ymax]

        if self.fractal_img is None:
            self.fractal_img = self.ax.imshow(fractal, extent=extent, origin='lower')
        else:
            self.fractal_img.set_data(fractal)
            self.fractal_img.set_extent(extent)

        # Clear previous markers & legend patches
        self.root_markers.clear()
        self.legend_patches.clear()

        for i, root in enumerate(self.roots):
            if self.mode == "iter":
                self._add_colorbar()
                # White markers with black edge for iter mode
                marker, = self.ax.plot(root.real, root.imag, 'o', color='w',
                                    markeredgecolor='k', markersize=8, picker=True)
                patch = Line2D([0], [0], marker='o', linestyle='', markersize=8,
                            markerfacecolor='w', markeredgecolor='k',
                            label=f"Root: {root:.2f}")
            else:
                # Colored markers and legend for other modes
                color = self.color_map.get(root, (1, 1, 1))
                marker, = self.ax.plot(root.real, root.imag, 'o', color=color,
                                    markeredgecolor='k', markersize=8, picker=True)
                patch = Line2D([0], [0], marker='o', linestyle='', markersize=8,
                            markerfacecolor=color, markeredgecolor='k',
                            label=f"Root: {root:.2f}")
            self.root_markers.append(marker)
            self.legend_patches.append(patch)

        if self.legend:
            self.legend.remove()
        self.legend = self.ax.legend(handles=self.legend_patches, loc='upper right', title="Roots")

        self.ax.set_title(f"Newton Fractal for $f(z) = z^{{{self.n}}} - 1$")
        self.ax.set_xlabel("Re(z)")
        self.ax.set_ylabel("Im(z)")
        self.ax.set_aspect("equal")

        self.fig.canvas.draw_idle()
        self._check_axes_limits()

    def _add_colorbar(self):
        # Remove any existing colorbar to prevent duplicates
        if self.colorbar:
            self.colorbar.remove()
            self.colorbar = None

        # Build an actual used color map from fractal (real used iteration range)
        used_iters = np.unique((self.norm_iters * self.real_max_iter).astype(int))
        if len(used_iters) <= 1:
            # If there's no meaningful iteration range, don't display a colorbar
            return

        # Create the colormap and scalar mappable
        colors = plt.get_cmap(self.mode_colormaps["iter"])(used_iters / self.real_max_iter)[:, :3]
        user_cmap = mcolors.ListedColormap(colors)
        norm = mcolors.Normalize(vmin=used_iters.min(), vmax=used_iters.max())
        sm = plt.cm.ScalarMappable(cmap=user_cmap, norm=norm)
        sm.set_array([])

        # --- TIE COLORBAR POSITION TO SPECIFIC COORDINATES ---
        cbar_ax_left = 0.8842
        cbar_ax_bottom = 0.1484
        cbar_ax_width = 0.0300
        cbar_ax_height = 0.7

        # Create a new axes explicitly for the colorbar using these fixed coordinates
        cbar_ax = self.fig.add_axes([cbar_ax_left, cbar_ax_bottom, cbar_ax_width, cbar_ax_height])

        # Draw the colorbar into this new, fixed axes using 'cax'
        self.colorbar = self.fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        # --- END TIE COLORBAR POSITION ---

        self.colorbar.set_label('Iteration Count')

        self.fig.canvas.draw_idle()

    def _on_pick(self, event):
        if self.mode == "iter":
            return
        # Detect if a root marker was clicked
        if event.artist in self.root_markers:
            idx = self.root_markers.index(event.artist)
            self._change_color(idx)

    def _on_click(self, event):
        # Detect clicks on legend text or markers
        if self.legend is None or self.mode == "iter":
            return
        # Check if click inside legend bbox
        legend_box = self.legend.get_window_extent()
        if not legend_box.contains(event.x, event.y):
            return

        for i, text in enumerate(self.legend.get_texts()):
            if text.get_window_extent().contains(event.x, event.y):
                self._change_color(i)
                return

        for i, handle in enumerate(self.legend.legend_handles):
            if handle.get_window_extent().contains(event.x, event.y):
                self._change_color(i)
                return
            
    def _on_colorbar_click(self, event):
        if self.mode != "iter" or not self.colorbar:
            return

        if not self.colorbar.ax.bbox.contains(event.x, event.y):
            return
        if self.iter_color_mode == "old" or self.iter_color_mode == "hist_2":
            start_rgb, _ = askcolor(title="Start color (low iteration)")
            if not start_rgb:
                return
            end_rgb, _ = askcolor(title="End color (high iteration)")
            if not end_rgb:
                return

            self._recolor_fractal(start_rgb=start_rgb, end_rgb=end_rgb)
        if self.iter_color_mode == "hist" or self.iter_color_mode == "hsv":
            user_colors = []
            for i in range(6):
                rgb, _ = askcolor(title=f"Choose gradient color {i + 1} (Cancel to stop)")
                if not rgb:
                    break
                user_colors.append(rgb)

            if len(user_colors) < 2:
                return  # Need at least 2 colors

            # Generate and apply new perceptual colormap
            self.color_map = self.custom_gradient_colormap(user_colors, brightness_curve="sin")
            self._recolor_fractal(cmap=self.color_map)

    def _change_color(self, index):
        root = self.roots[index]
        rgb, _ = askcolor(title=f"Choose new color for Root {index + 1}")
        if rgb:
            color = tuple(c / 255 for c in rgb)
            self.color_map[root] = color
            self._recolor_fractal()

    def _update_iter_colorbar(self, cmap, max_iter):
        norm = mcolors.Normalize(vmin=0, vmax=max_iter)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_cmap(cmap)
        sm.set_norm(norm)
        self.colorbar.update_normal(sm)

        self.fig.canvas.draw_idle()

    def _recolor_fractal(self, start_rgb=None, end_rgb=None, cmap=None):
        if self.mode == "iter" and not hasattr(self, "norm_iters"):
            return  # No fractal yet

        elif not hasattr(self, "last_root_ids"):
            return  # No cached data, fallback to full update

        arr, shape = self._generate_meshgrid()
        img = np.zeros((arr.shape[0], 3))

        if self.mode == "iter":
            if self.iter_color_mode == "old":
                start = np.array(start_rgb) / 255
                end = np.array(end_rgb) / 255

                n_colors = 256
                gradient_colors = np.linspace(start, end, n_colors)
                custom_cmap = mcolors.ListedColormap(gradient_colors)

                # Normalize iterations and map to gradient
                norm = mcolors.Normalize(vmin=0, vmax=1)
                mapped_colors = custom_cmap(norm(self.norm_iters.flatten()))[:, :3]

                # Apply brightness scaling
                img = mapped_colors * self.mode_brightness["iter"]

                self._update_iter_colorbar(custom_cmap, self.real_max_iter)
            elif self.iter_color_mode == "hist_2":
                start = np.array(start_rgb) / 255
                end = np.array(end_rgb) / 255

                # Build gradient
                n_colors = 256
                gradient_colors = np.linspace(start, end, n_colors)
                custom_cmap = mcolors.ListedColormap(gradient_colors)

                # HISTOGRAM EQUALIZATION 
                iter_flat = (self.norm_iters.flatten() * self.real_max_iter).astype(int)
                hist, bin_edges = np.histogram(iter_flat, bins=n_colors, range=(0, self.real_max_iter))
                cdf = np.cumsum(hist)
                cdf = cdf / cdf[-1]  # normalize to 0–1

                # Map each pixel’s iteration count to equalized color index
                cdf_mapped = np.interp(iter_flat, bin_edges[:-1], cdf)

                # Use color map with adjusted (equalized) values
                mapped_colors = custom_cmap(cdf_mapped)[:, :3]
                img = mapped_colors * self.mode_brightness["iter"]

                self._update_iter_colorbar(custom_cmap, self.real_max_iter)
            elif self.iter_color_mode == "hsv":
                if cmap is None:
                    cmap = plt.get_cmap(self.mode_colormaps["iter"])
                mapped_colors = cmap(self.norm_iters.flatten())[:, :3]
                img = mapped_colors * self.mode_brightness["iter"]
                self._update_iter_colorbar(cmap, self.real_max_iter)
            elif self.iter_color_mode == "hist":
                if cmap is None:
                    return  # cmap must be provided

                n_colors = cmap.N
                iter_flat = (self.norm_iters.flatten() * self.real_max_iter).astype(int)
                hist, bin_edges = np.histogram(iter_flat, bins=n_colors, range=(0, self.real_max_iter))
                cdf = np.cumsum(hist)
                cdf = cdf / cdf[-1]  # normalize to 0–1
                cdf_mapped = np.interp(iter_flat, bin_edges[:-1], cdf)

                mapped_colors = cmap(cdf_mapped)[:, :3]
                img = mapped_colors * self.mode_brightness["iter"]

                self._update_iter_colorbar(cmap, self.real_max_iter)


        else:
            # Modes: 'root' or 'both'
            for i, root in enumerate(self.roots, 1):
                mask = self.last_root_ids == i
                base_color = self.color_map.get(root, (1, 1, 1))
                if self.mode == "root":
                    img[mask] = np.array(base_color)[:3] * self.mode_brightness["root"]
                elif self.mode == "both":
                    img[mask] = base_color * (self.last_brightness[mask, np.newaxis] * self.mode_brightness["both"])

        # Update image
        self.fractal_img.set_data(img.reshape(*shape, 3))

        # Update markers and legend
        for i, root in enumerate(self.roots):
            if self.mode == "iter":
                self.root_markers[i].set_color('w')
                self.root_markers[i].set_markeredgecolor('k')
                self.legend.legend_handles[i].set_markerfacecolor('w')
                self.legend.legend_handles[i].set_markeredgecolor('k')
            else:
                self.root_markers[i].set_color(self.color_map[root])
                self.root_markers[i].set_markeredgecolor('k')
                self.legend.legend_handles[i].set_markerfacecolor(self.color_map[root])
                self.legend.legend_handles[i].set_markeredgecolor('k')

        self.fig.canvas.draw_idle()

    def custom_gradient_colormap(self, colors_rgb, brightness_curve="sin", v_start=0.8, v_end=1.0):
        """
        Create a perceptually modulated colormap using HSV interpolation + brightness curve.

        colors_rgb: list of RGB tuples [(r, g, b), ...] in 0–255
        brightness_curve: 'sin', 'exp', or 'linear'
        v_start: minimum brightness at start (0 to 1)
        v_end: minimum brightness at end (0 to 1)
        """
        from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

        # Convert to numpy 0–1
        rgb = np.array(colors_rgb) / 255.0
        hsv = rgb_to_hsv(rgb)

        n_colors = 256
        positions = np.linspace(0, 1, len(hsv))
        interp_positions = np.linspace(0, 1, n_colors)

        # Interpolate HSV channels
        h = np.interp(interp_positions, positions, hsv[:, 0])
        s = np.interp(interp_positions, positions, hsv[:, 1])
        v = np.interp(interp_positions, positions, hsv[:, 2])

        # Brightness modulation shape
        if brightness_curve == "sin":
            shape = np.sin(np.linspace(0, np.pi, n_colors))
        elif brightness_curve == "exp":
            x = np.linspace(0, 1, n_colors)
            shape = np.exp(-3 * x)
        else:
            shape = np.linspace(1, 0.5, n_colors)

        # Normalize shape and scale to desired v_start/v_end
        shape = shape - shape.min()
        shape = shape / shape.max()  # now between 0–1
        brightness = v_start + (v_end - v_start) * shape

        # Apply brightness to V channel
        v = np.clip(v * brightness, 0, 1)
        hsv_mod = np.stack([h, s, v], axis=1)

        rgb_mod = hsv_to_rgb(hsv_mod)
        return mcolors.ListedColormap(rgb_mod)

    def _regenerate_fractal(self, xlim, ylim):
        # Map the current axes limits to fractal bounds
        self.xmin, self.xmax = xlim
        self.ymin, self.ymax = ylim

        # Regenerate fractal with these new bounds
        data = self.generate_fractal()

        # Update image data
        self.fractal_img.set_data(data)
        self.fractal_img.set_extent((self.xmin, self.xmax, self.ymin, self.ymax))

        # Optionally adjust color scale
        self._recolor_fractal(cmap=self.color_map)

        # Update colorbar if needed
        if self.mode == "iter":
            self._update_iter_colorbar(self.color_map, self.real_max_iter)

        # Redraw canvas
        self.fig.canvas.draw_idle()

    def _check_axes_limits(self):
        new_xlim = self.ax.get_xlim()
        new_ylim = self.ax.get_ylim()

        if new_xlim[0]!=self.xmin or new_xlim[1]!=self.xmax or new_ylim[0]!=self.ymin or new_ylim[1]!=self.ymax:
            self._last_xlim = new_xlim
            self._last_ylim = new_ylim
            self._regenerate_fractal(new_xlim, new_ylim)

        # Call this method again after 200ms (or tune as needed)
        self.tk_root.after(200, self._check_axes_limits)


NewtonFractalInteractive()