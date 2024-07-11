import time
import numpy as np
import os
import json
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns


def generate_plots(params):
    """
    """
    print('Begin generating plots\n')
    start = time.time()

    # Load Experiment Mode
    gradient = params['gradient']
    sampling = params['sampling']
    # Load Experiment Parameters
    max_iterations = params['max_iterations']
    # environment
    fbeta = params['fbeta']
    betas = params['betas']
    fgamma = params['fgamma']
    gammas = params['gammas']
    # perormative prediction parameters
    max_iterations = params['max_iterations']
    iterations_printed = params['iterations_printed']
    flamda = params['flamda']
    lamdas = params['lamdas']
    freg = params['freg']
    regs = params['regs']
    # gradient parameters
    feta = params['feta']
    etas = params['etas']
    # sampling parameters
    fn_sample = params['fn_sample']
    n_samples = params['n_samples']
    # policy gradient
    policy_gradient = params['policy_gradient']
    # unregularized objective
    unregularized_obj = params['unregularized_obj']

    # Load Output
    with open(f'data/outputs.json', 'r') as f:
        output = json.load(f)

    if not gradient and not sampling and not policy_gradient and not unregularized_obj:
        # iterate lamdas
        if lamdas:
            if not os.path.exists(f'figures/lambdas'):
                os.mkdir(f'figures/lambdas')

            lst = sorted([d for d in output if d['beta'] == fbeta and d['gamma'] == fgamma and d['reg'] == freg], key=lambda d: d['lamda'])

            fig, ax = plt.subplots()
            # plot
            for d in lst:
                lamda = d['lamda']
                plt.plot(list(range(max_iterations)), d['d_diff'], linewidth=2, label=f'$\lambda$={lamda}')
            plt.yscale('log')
            plt.xlabel('Iteration t', fontsize=30)
            plt.ylabel('$c_t \cdot ||d_{t+1} - d_t||_2$', fontsize=30)
            plt.tick_params(labelsize=20)
            plt.tight_layout()
            axes = plt.gca()
            axes.tick_params(bottom=True, top=False, left=True, right=False)
            axes.spines['bottom'].set_color('0')
            axes.spines['top'].set_color('0')
            axes.spines['right'].set_color('0')
            axes.spines['left'].set_color('0')
            axes.set_facecolor('w')
            fig.savefig(f"figures/lambdas/beta={fbeta}_gamma={fgamma}_reg={freg}.pdf", bbox_inches = 'tight')
            # legend
            legend = plt.legend(
                    ncol=3, fancybox=True, facecolor="white",
                    shadow=True, fontsize=20
                )
            fig.canvas.draw()
            legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
            legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
            legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width, legend_bbox.height))
            legend_squared = legend_ax.legend(
                *ax.get_legend_handles_labels(), 
                bbox_to_anchor=(0, 0, 1, 1),
                bbox_transform=legend_fig.transFigure,
                frameon=True,
                facecolor="white",
                fancybox=True,
                shadow=True,
                ncol=3,
                fontsize=20,
            )
            legend_ax.axis('off')
            legend_fig.savefig("figures/lambdas/lambdas_legend.pdf",  
                                bbox_inches='tight',
                                bbox_extra_artists=[legend_squared]
            )
            plt.close(legend_fig)
            plt.close(fig)

        # iterate betas
        if betas:
            if not os.path.exists(f'figures/betas'):
                os.mkdir(f'figures/betas')

            lst = sorted([d for d in output if d['lamda'] == flamda and d['gamma'] == fgamma and d['reg'] == freg], key=lambda d: d['beta'])

            fig, ax = plt.subplots()
            # plot
            for d in lst:
                beta = d['beta']
                plt.plot(list(range(max_iterations)), d['d_diff'], linewidth=2, label=f'$\\beta$={beta}')
            plt.yscale('log')
            plt.xlabel('Iteration t', fontsize=30)
            plt.ylabel('$c_t \cdot ||d_{t+1} - d_t||_2$', fontsize=30)
            plt.tick_params(labelsize=20)
            plt.tight_layout()
            axes = plt.gca()
            axes.tick_params(bottom=True, top=False, left=True, right=False)
            axes.spines['bottom'].set_color('0')
            axes.spines['top'].set_color('0')
            axes.spines['right'].set_color('0')
            axes.spines['left'].set_color('0')
            axes.set_facecolor('w')
            fig.savefig(f"figures/betas/lambda={flamda}_gamma={fgamma}_reg={freg}.pdf", bbox_inches = 'tight')
            # legend
            legend = plt.legend(
                    ncol=3, fancybox=True, facecolor="white",
                    shadow=True, fontsize=20
                )
            fig.canvas.draw()
            legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
            legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
            legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width, legend_bbox.height))
            legend_squared = legend_ax.legend(
                *ax.get_legend_handles_labels(), 
                bbox_to_anchor=(0, 0, 1, 1),
                bbox_transform=legend_fig.transFigure,
                frameon=True,
                facecolor="white",
                fancybox=True,
                shadow=True,
                ncol=3,
                fontsize=20,
            )
            legend_ax.axis('off')
            legend_fig.savefig("figures/betas/betas_legend.pdf",  
                                bbox_inches='tight',
                                bbox_extra_artists=[legend_squared]
            )
            plt.close(legend_fig)
            plt.close(fig)

        # iterate regs
        if regs:
            if not os.path.exists(f'figures/regs'):
                os.mkdir(f'figures/regs')

            lst = sorted([d for d in output if d['beta'] == fbeta and d['lamda'] == flamda and d['gamma'] == fgamma], key=lambda d: d['reg'])

            fig, ax = plt.subplots()
            # plot
            for d in lst:
                reg = d['reg']
                plt.plot(list(range(max_iterations)), d['d_diff'], linewidth=2, label=f'reg={reg}')
            plt.yscale('log')
            plt.xlabel('Iteration t', fontsize=30)
            plt.ylabel('$c_t \cdot ||d_{t+1} - d_t||_2$', fontsize=30)
            plt.tick_params(labelsize=20)
            plt.tight_layout()
            axes = plt.gca()
            axes.tick_params(bottom=True, top=False, left=True, right=False)
            axes.spines['bottom'].set_color('0')
            axes.spines['top'].set_color('0')
            axes.spines['right'].set_color('0')
            axes.spines['left'].set_color('0')
            axes.set_facecolor('w')
            fig.savefig(f"figures/regs/beta={fbeta}_lambda={flamda}_gamma={fgamma}.pdf", bbox_inches = 'tight')
            # legend
            legend = plt.legend(
                    ncol=3, fancybox=True, facecolor="white",
                    shadow=True, fontsize=20
                )
            fig.canvas.draw()
            legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
            legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
            legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width, legend_bbox.height))
            legend_squared = legend_ax.legend(
                *ax.get_legend_handles_labels(), 
                bbox_to_anchor=(0, 0, 1, 1),
                bbox_transform=legend_fig.transFigure,
                frameon=True,
                facecolor="white",
                fancybox=True,
                shadow=True,
                ncol=3,
                fontsize=20,
            )
            legend_ax.axis('off')
            legend_fig.savefig("figures/regs/regs_legend.pdf",  
                                bbox_inches='tight',
                                bbox_extra_artists=[legend_squared]
            )
            plt.close(legend_fig)
            plt.close(fig)

        # iterate gammas and lamdas
        if gammas and lamdas:
            if not os.path.exists(f'figures/gammas_lambdas'):
                os.mkdir(f'figures/gammas_lambdas')

            lst = sorted([d for d in output if d['beta'] == fbeta and d['reg'] == freg], key=lambda d: d['lamda'])

            fig, ax = plt.subplots()
            # plot
            for gamma in gammas:
                for d in [d for d in lst if d['gamma'] == gamma]:
                    lamda = d['lamda']
                    plt.plot(list(range(max_iterations)), d['d_diff'], linewidth=2, label=f'$\lambda$={lamda}')
                plt.yscale('log')
                plt.xlabel('Iteration t', fontsize=30)
                plt.ylabel('$c_t \cdot ||d_{t+1} - d_t||_2$', fontsize=30)
                plt.tick_params(labelsize=20)
                plt.tight_layout()
                axes = plt.gca()
                axes.tick_params(bottom=True, top=False, left=True, right=False)
                axes.spines['bottom'].set_color('0')
                axes.spines['top'].set_color('0')
                axes.spines['right'].set_color('0')
                axes.spines['left'].set_color('0')
                axes.set_facecolor('w')
                fig.savefig(f"figures/gammas_lambdas/beta={fbeta}_gamma={gamma}_reg={freg}.pdf", bbox_inches = 'tight')
                # legend
                legend = plt.legend(
                        ncol=3, fancybox=True, facecolor="white",
                        shadow=True, fontsize=20
                    )
                fig.canvas.draw()
                legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
                legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
                legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width, legend_bbox.height))
                legend_squared = legend_ax.legend(
                    *ax.get_legend_handles_labels(), 
                    bbox_to_anchor=(0, 0, 1, 1),
                    bbox_transform=legend_fig.transFigure,
                    frameon=True,
                    facecolor="white",
                    fancybox=True,
                    shadow=True,
                    ncol=3,
                    fontsize=20,
                )
                legend_ax.axis('off')
                legend_fig.savefig("figures/gammas_lambdas/lambdas_legend.pdf",  
                                    bbox_inches='tight',
                                    bbox_extra_artists=[legend_squared]
                )
                plt.close(legend_fig)
                plt.close(fig)

    if gradient and etas:
        # iterate etas
        if not os.path.exists(f'figures/etas'):
            os.mkdir(f'figures/etas')

        if sampling:
            lst = sorted([d for d in output if d['n_sample'] == fn_sample], key=lambda d: d['eta'])
        else:
            lst = sorted([d for d in output], key=lambda d: d['eta'])

        fig, ax = plt.subplots()
        # plot
        for d in lst:
            eta = d['eta']
            if sampling:
                plt.plot(list(range(max_iterations)), d['d_diff_mean'], linewidth=2, label=f'$\eta$={eta}')
                plt.fill_between(list(range(max_iterations)), [d['d_diff_mean'][i] - d['d_diff_std'][i] for i in range(max_iterations)], [d['d_diff_mean'][i] + d['d_diff_std'][i] for i in range(max_iterations)])
            else:
                plt.plot(list(range(max_iterations)), d['d_diff'], linewidth=2, label=f'$\eta$={eta}')
        plt.yscale('log')
        plt.xlabel('Iteration t', fontsize=30)
        plt.ylabel('$c_t \cdot ||d_{t+1} - d_t||_2$', fontsize=30)
        plt.tick_params(labelsize=20)
        plt.tight_layout()
        axes = plt.gca()
        axes.tick_params(bottom=True, top=False, left=True, right=False)
        axes.spines['bottom'].set_color('0')
        axes.spines['top'].set_color('0')
        axes.spines['right'].set_color('0')
        axes.spines['left'].set_color('0')
        axes.set_facecolor('w')
        if sampling:
            fig.savefig(f"figures/etas/beta={fbeta}_lambda={flamda}_gamma={fgamma}_n_sample={fn_sample}.pdf", bbox_inches = 'tight')
        else:
            fig.savefig(f"figures/etas/beta={fbeta}_lambda={flamda}_gamma={fgamma}.pdf", bbox_inches = 'tight')
        # legend
        legend = plt.legend(
                ncol=3, fancybox=True, facecolor="white",
                shadow=True, fontsize=20
            )
        fig.canvas.draw()
        legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
        legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
        legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width, legend_bbox.height))
        legend_squared = legend_ax.legend(
            *ax.get_legend_handles_labels(), 
            bbox_to_anchor=(0, 0, 1, 1),
            bbox_transform=legend_fig.transFigure,
            frameon=True,
            facecolor="white",
            fancybox=True,
            shadow=True,
            ncol=3,
            fontsize=20,
        )
        legend_ax.axis('off')
        legend_fig.savefig("figures/etas/etas_legend.pdf",  
                            bbox_inches='tight',
                            bbox_extra_artists=[legend_squared]
        )
        plt.close(legend_fig)
        plt.close(fig)

        # suboptimality gap
        fig, ax = plt.subplots()
        # plot
        for d in lst:
            eta = d['eta']
            if sampling:
                plt.plot(list(range(max_iterations)), d['sub_gap_mean'], linewidth=2, label=f'$\eta$={eta}')
                plt.fill_between(list(range(max_iterations)), [d['sub_gap_mean'][i] - d['sub_gap_std'][i] for i in range(max_iterations)], [d['sub_gap_mean'][i] + d['sub_gap_std'][i] for i in range(max_iterations)], alpha=0.5)
            else:
                plt.plot(list(range(max_iterations)), d['sub_gap'], linewidth=2, label=f'$\eta$={eta}')
        plt.yscale('log')
        plt.xlabel('Iteration t', fontsize=30)
        plt.ylabel('Suboptimality Gap', fontsize=30)
        plt.tick_params(labelsize=20)
        plt.tight_layout()
        axes = plt.gca()
        axes.tick_params(bottom=True, top=False, left=True, right=False)
        axes.spines['bottom'].set_color('0')
        axes.spines['top'].set_color('0')
        axes.spines['right'].set_color('0')
        axes.spines['left'].set_color('0')
        axes.set_facecolor('w')
        if sampling:
            fig.savefig(f"figures/etas/sub_beta={fbeta}_lambda={flamda}_gamma={fgamma}_n_sample={fn_sample}.pdf", bbox_inches = 'tight')
        else:
            fig.savefig(f"figures/etas/sub_beta={fbeta}_lambda={flamda}_gamma={fgamma}.pdf", bbox_inches = 'tight')

    if sampling and n_samples:
        # iterate n_samples
        if not os.path.exists(f'figures/n_samples'):
            os.mkdir(f'figures/n_samples')

        if gradient:
            lst = sorted([d for d in output if d['eta'] == feta], key=lambda d: d['n_sample'])
        else:
            lst = sorted([d for d in output], key=lambda d: d['n_sample'])

        fig, ax = plt.subplots()
        # plot
        for d in lst:
            n_sample = d['n_sample']
            plt.plot(list(range(max_iterations)), d['d_diff_mean'], linewidth=2, label=f'm={n_sample}')
            plt.fill_between(list(range(max_iterations)), [d['d_diff_mean'][i] - d['d_diff_std'][i] for i in range(max_iterations)], [d['d_diff_mean'][i] + d['d_diff_std'][i] for i in range(max_iterations)], alpha=0.5)
        # plt.yscale('log')
        plt.xlabel('Iteration t', fontsize=30)
        plt.ylabel('$c_t \cdot ||d_{t+1} - d_t||_2$', fontsize=30)
        plt.tick_params(labelsize=20)
        plt.tight_layout()
        axes = plt.gca()
        axes.tick_params(bottom=True, top=False, left=True, right=False)
        axes.spines['bottom'].set_color('0')
        axes.spines['top'].set_color('0')
        axes.spines['right'].set_color('0')
        axes.spines['left'].set_color('0')
        axes.set_facecolor('w')
        if gradient:
            fig.savefig(f"figures/n_samples/beta={fbeta}_lambda={flamda}_gamma={fgamma}_reg={freg}_eta={feta}.pdf", bbox_inches = 'tight')
        else:
            fig.savefig(f"figures/n_samples/beta={fbeta}_lambda={flamda}_gamma={fgamma}_reg={freg}.pdf", bbox_inches = 'tight')
        # legend
        legend = plt.legend(
                ncol=3, fancybox=True, facecolor="white",
                shadow=True, fontsize=20
            )
        fig.canvas.draw()
        legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
        legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
        legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width, legend_bbox.height))
        legend_squared = legend_ax.legend(
            *ax.get_legend_handles_labels(), 
            bbox_to_anchor=(0, 0, 1, 1),
            bbox_transform=legend_fig.transFigure,
            frameon=True,
            facecolor="white",
            fancybox=True,
            shadow=True,
            ncol=3,
            fontsize=20,
        )
        legend_ax.axis('off')
        legend_fig.savefig("figures/n_samples/n_samples_legend.pdf",  
                            bbox_inches='tight',
                            bbox_extra_artists=[legend_squared]
        )
        plt.close(legend_fig)
        plt.close(fig)

        # plot trajectory lengths (ONLY for the first seed)
        if not os.path.exists(f'figures/additional_diagnostics'):
            os.mkdir(f'figures/additional_diagnostics')
        trajectory_fig, trajectory_ax = plt.subplots()
        for d in lst:
            n_sample = d['n_sample']
            plt.hist(d['trajectory_length'], bins = 100, alpha = 0.5, label=f'm={n_sample}')

        plt.xlabel('Trajectory length', fontsize=30)
        plt.ylabel('Count', fontsize=30)
        plt.tick_params(labelsize=20)
        plt.tight_layout()
        axes_trajectory = plt.gca()
        axes_trajectory.tick_params(bottom=True, top=False, left=True, right=False)
        axes_trajectory.spines['bottom'].set_color('0')
        axes_trajectory.spines['top'].set_color('0')
        axes_trajectory.spines['right'].set_color('0')
        axes_trajectory.spines['left'].set_color('0')
        axes_trajectory.set_facecolor('w')

        trajectory_fig.savefig(f"figures/additional_diagnostics/trajectory_length.pdf", bbox_inches = 'tight')

        # legend
        trajectory_legend = plt.legend(
                ncol=3, fancybox=True, facecolor="white",
                shadow=True, fontsize=20
            )
        trajectory_fig.canvas.draw()
        trajectory_legend_bbox = trajectory_legend.get_tightbbox(trajectory_fig.canvas.get_renderer())
        trajectory_legend_bbox = trajectory_legend_bbox.transformed(trajectory_fig.dpi_scale_trans.inverted())
        trajectory_legend_fig, trajectory_legend_ax = plt.subplots(figsize=(trajectory_legend_bbox.width, trajectory_legend_bbox.height))
        trajectory_legend_squared = trajectory_legend_ax.legend(
            *ax.get_legend_handles_labels(), 
            bbox_to_anchor=(0, 0, 1, 1),
            bbox_transform=trajectory_legend_fig.transFigure,
            frameon=True,
            facecolor="white",
            fancybox=True,
            shadow=True,
            ncol=3,
            fontsize=20,
        )
        trajectory_legend_ax.axis('off')
        trajectory_legend_fig.savefig("figures/additional_diagnostics/trajectory_length_legend.pdf",  
                            bbox_inches='tight',
                            bbox_extra_artists=[trajectory_legend_squared]
        )
        plt.close(trajectory_legend_fig)
        plt.close(trajectory_fig)

        # plot state space coverage per iteration (ONLY for the first seed)        
        for d in lst:
            # subplots to display the first 10 iterations
            sp_fig, sp_ax = plt.subplots(2,5, figsize=(20, 8))
            n_sample = d['n_sample']
            # Create a colormap: red for visited (True), white for not visited (False)
            cmap = ListedColormap(['white', 'red'])
            bounds = [0, 0.5, 1]
            norm = BoundaryNorm(bounds, cmap.N)
            for i in range(0,10):
                # Reshape the list to an 8x8 numpy array
                visited_array = np.array(d['state_space_coverage_iteration_grid'][iterations_printed[i]]).reshape((8, 8))
                # Determine the position of the subplot
                ax = sp_ax[i // 5, i % 5]
                ax.imshow(visited_array, cmap=cmap, norm=norm)
                ax.set_title(f'n_sample:{n_sample}, iteration: {iterations_printed[i]}')
                # Add grid lines
                ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
                ax.grid(which='major', color='black', linestyle='-', linewidth=2)
                ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
                
                # Remove tick labels
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            
            plt.tight_layout()
            plt.savefig(f"figures/additional_diagnostics/statespace_coverage_iteration_n_sample={d['n_sample']}.pdf", bbox_inches = 'tight')
            plt.close(sp_fig)

        # plot state space coverage per iteration (ONLY for the first seed)
        for d in lst:
            # subplots to display the first 10 iterations
            sp_fig, sp_ax = plt.subplots(2,5, figsize=(20, 8))
            n_sample = d['n_sample']
            for i in range(0,10):
                # Reshape the list to an 8x8 numpy array
                visited_array = np.array(d['state_visitation_counts_iteration'][iterations_printed[i]]).reshape((8, 8))
                # Determine the position of the subplot
                ax = sp_ax[i // 5, i % 5]
                sns.heatmap(visited_array, ax=ax, annot=True, cmap='crest', cbar=True, xticklabels=False, yticklabels=False)
                ax.set_title(f'n_sample:{n_sample}, iteration: {iterations_printed[i]}')
            
            plt.tight_layout()
            plt.savefig(f"figures/additional_diagnostics/statespace_coverage_heatmap_n_sample={d['n_sample']}.pdf", bbox_inches = 'tight')
            plt.close(sp_fig)

        # suboptimality gap
        if gradient:
            fig, ax = plt.subplots()
            # plot
            for d in lst:
                n_sample = d['n_sample']
                plt.plot(list(range(max_iterations)), d['sub_gap_mean'], linewidth=2, label=f'm={n_sample}')
                plt.fill_between(list(range(max_iterations)), [d['sub_gap_mean'][i] - d['sub_gap_std'][i] for i in range(max_iterations)], [d['sub_gap_mean'][i] + d['sub_gap_std'][i] for i in range(max_iterations)], alpha=0.5)
            plt.xlabel('Iteration t', fontsize=30)
            plt.ylabel('Suboptimality Gap', fontsize=30)
            plt.tick_params(labelsize=20)
            plt.tight_layout()
            axes = plt.gca()
            axes.tick_params(bottom=True, top=False, left=True, right=False)
            axes.spines['bottom'].set_color('0')
            axes.spines['top'].set_color('0')
            axes.spines['right'].set_color('0')
            axes.spines['left'].set_color('0')
            axes.set_facecolor('w')
            if gradient:
                fig.savefig(f"figures/n_samples/sub_beta={fbeta}_lambda={flamda}_gamma={fgamma}_reg={freg}_eta={feta}.pdf", bbox_inches = 'tight')
            else:
                fig.savefig(f"figures/n_samples/sub_beta={fbeta}_lambda={flamda}_gamma={fgamma}_reg={freg}.pdf", bbox_inches = 'tight')
            plt.close()

    if policy_gradient:
        # iterate nus
        if not os.path.exists(f'figures/nus'):
            os.mkdir(f'figures/nus')

        lst = sorted([d for d in output], key=lambda d: d['nu'])

        fig, ax = plt.subplots()
        # plot
        for d in lst:
            nu = d['nu']
            plt.plot(list(range(max_iterations)), d['d_diff'], linewidth=2, label=f'$\\nu$={nu}')
        plt.yscale('log')
        plt.xlabel('Iteration t', fontsize=30)
        plt.ylabel('$c_t \cdot ||d_{t+1} - d_t||_2$', fontsize=30)
        plt.tick_params(labelsize=20)
        plt.tight_layout()
        axes = plt.gca()
        axes.tick_params(bottom=True, top=False, left=True, right=False)
        axes.spines['bottom'].set_color('0')
        axes.spines['top'].set_color('0')
        axes.spines['right'].set_color('0')
        axes.spines['left'].set_color('0')
        axes.set_facecolor('w')
        fig.savefig(f"figures/nus/beta={fbeta}_lambda={flamda}_gamma={fgamma}_eta={feta}.pdf", bbox_inches = 'tight')
        # legend
        legend = plt.legend(
                ncol=3, fancybox=True, facecolor="white",
                shadow=True, fontsize=20
            )
        fig.canvas.draw()
        legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
        legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
        legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width, legend_bbox.height))
        legend_squared = legend_ax.legend(
            *ax.get_legend_handles_labels(), 
            bbox_to_anchor=(0, 0, 1, 1),
            bbox_transform=legend_fig.transFigure,
            frameon=True,
            facecolor="white",
            fancybox=True,
            shadow=True,
            ncol=3,
            fontsize=20,
        )
        legend_ax.axis('off')
        legend_fig.savefig("figures/nus/nus_legend.pdf",  
                            bbox_inches='tight',
                            bbox_extra_artists=[legend_squared]
        )
        plt.close(legend_fig)
        plt.close(fig)

        # suboptimality gap
        fig, ax = plt.subplots()
        # plot
        for d in lst:
            nu = d['nu']
            plt.plot(list(range(max_iterations)), d['sub_gap'], linewidth=2, label=f'$\\nu$={nu}')
        plt.xlabel('Iteration t', fontsize=30)
        plt.ylabel('Suboptimality Gap', fontsize=30)
        plt.ylim(bottom=0)
        plt.tick_params(labelsize=20)
        plt.tight_layout()
        axes = plt.gca()
        axes.tick_params(bottom=True, top=False, left=True, right=False)
        axes.spines['bottom'].set_color('0')
        axes.spines['top'].set_color('0')
        axes.spines['right'].set_color('0')
        axes.spines['left'].set_color('0')
        axes.set_facecolor('w')
        fig.savefig(f"figures/nus/sub_beta={fbeta}_lambda={flamda}_gamma={fgamma}_eta={feta}.pdf", bbox_inches = 'tight')
    
    if unregularized_obj and lamdas:
        # iterate lamdas
        if not os.path.exists(f'figures/unreg_obj'):
            os.mkdir(f'figures/unreg_obj')

        lst = sorted([d for d in output], key=lambda d: d['lamda'])

        fig, ax = plt.subplots()
        # plot
        for d in lst:
            lamda = d['lamda']
            plt.plot(list(range(max_iterations)), d['d_diff'], linewidth=2, label=f'$\lambda$={lamda}')
        plt.yscale('log')
        plt.xlabel('Iteration t', fontsize=30)
        plt.ylabel('$c_t \cdot ||d_{t+1} - d_t||_2$', fontsize=30)
        plt.tick_params(labelsize=20)
        plt.tight_layout()
        axes = plt.gca()
        axes.tick_params(bottom=True, top=False, left=True, right=False)
        axes.spines['bottom'].set_color('0')
        axes.spines['top'].set_color('0')
        axes.spines['right'].set_color('0')
        axes.spines['left'].set_color('0')
        axes.set_facecolor('w')
        fig.savefig(f"figures/unreg_obj/beta={fbeta}_gamma={fgamma}.pdf", bbox_inches = 'tight')
        # legend
        legend = plt.legend(
                ncol=3, fancybox=True, facecolor="white",
                shadow=True, fontsize=20
            )
        fig.canvas.draw()
        legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
        legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
        legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width, legend_bbox.height))
        legend_squared = legend_ax.legend(
            *ax.get_legend_handles_labels(), 
            bbox_to_anchor=(0, 0, 1, 1),
            bbox_transform=legend_fig.transFigure,
            frameon=True,
            facecolor="white",
            fancybox=True,
            shadow=True,
            ncol=3,
            fontsize=20,
        )
        legend_ax.axis('off')
        legend_fig.savefig("figures/unreg_obj/lambdas_legend.pdf",  
                            bbox_inches='tight',
                            bbox_extra_artists=[legend_squared]
        )
        plt.close(legend_fig)
        plt.close(fig)

        # suboptimality gap
        fig, ax = plt.subplots()
        # plot
        for d in lst:
            lamda = d['lamda']
            plt.plot(list(range(max_iterations)), d['sub_gap'], linewidth=2, label=f'$\lambda$={lamda}')
        plt.xlabel('Iteration t', fontsize=30)
        plt.ylabel('Suboptimality Gap', fontsize=30)
        plt.ylim(bottom=0)
        plt.tick_params(labelsize=20)
        plt.tight_layout()
        axes = plt.gca()
        axes.tick_params(bottom=True, top=False, left=True, right=False)
        axes.spines['bottom'].set_color('0')
        axes.spines['top'].set_color('0')
        axes.spines['right'].set_color('0')
        axes.spines['left'].set_color('0')
        axes.set_facecolor('w')
        fig.savefig(f"figures/unreg_obj/sub_beta={fbeta}_gamma={fgamma}.pdf", bbox_inches = 'tight')

    end = time.time()
    print(f'Time: {end - start}')
    print('Finish generating plots\n')

    return