import os
import itertools
import numpy as np
import glob
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

from tqdm import *

def load_data():

  prefix = 'overall_perf'

  arch = 'large_atari'

  base_path = '../results'

  domains = ['sokoban']

  algos = [ 'ddqn-uer', 
            'ddqn-per', 
            'ddqn-ebu', 
            'ddqn-discor-uer', 
            'ddqn-gameover-er-mixin-0.1-rs']

  algo_legend = { 'ddqn-uer': 'UER',
                  'ddqn-per': 'PER',
                  'ddqn-ebu': 'EBU',
                  'ddqn-discor-uer': 'DisCor',
                  'ddqn-gameover-er-mixin-0.1-rs': 'TER' }

  envs = [
    'Sokoban-Push_5x5_1_120',
    'Sokoban-Push_6x6_1_120',
    'Sokoban-Push_7x7_1_120',
    'Sokoban-Push_6x6_3_120',
    'Sokoban-Push_5x5_2_120',
    'Sokoban-Push_6x6_2_120',
    'Sokoban-Push_7x7_2_120',
  ]

  x_maxs = {
      'Sokoban-Push_5x5_1_120': 1e6,
      'Sokoban-Push_6x6_1_120': 1e6,
      'Sokoban-Push_7x7_1_120': 6e6,

      'Sokoban-Push_5x5_2_120': 1e6,
      'Sokoban-Push_6x6_2_120': 5e6,
      'Sokoban-Push_7x7_2_120': 5e6,

      'Sokoban-Push_6x6_3_120': 10e6,
  }

  rmaxs = {
      'Sokoban-Push_5x5_1_120': 11,
      'Sokoban-Push_6x6_1_120': 11,
      'Sokoban-Push_7x7_1_120': 11,

      'Sokoban-Push_5x5_2_120': 12,
      'Sokoban-Push_6x6_2_120': 12,
      'Sokoban-Push_7x7_2_120': 12,

      'Sokoban-Push_6x6_3_120': 13,
  }

  rmins = {
      'Sokoban-Push_5x5_1_120': -11,
      'Sokoban-Push_6x6_1_120': -11,
      'Sokoban-Push_7x7_1_120': -11,

      'Sokoban-Push_5x5_2_120': -12,
      'Sokoban-Push_6x6_2_120': -12,
      'Sokoban-Push_7x7_2_120': -12,

      'Sokoban-Push_6x6_3_120': -13,
  }
  
  all_dfs = []
  for domain, algo, env in tqdm(itertools.product(domains, algos, envs)):
      log_base_path = os.path.join(base_path, prefix, domain, algo, arch, env)
      for log_path in glob.glob(f'{log_base_path}/*'):      
        try:
          with open(os.path.join(log_path, 'scores.txt'), 'r') as logfile:
            df = pd.read_csv(logfile, sep='\t')
            df['normalized_mean'] = (df['mean'] - rmins.get(env, 0.0)) / (rmaxs.get(env, 1.0) - rmins.get(env, 0.0))
            df['env'] = env
            df['algo'] = algo_legend[algo]
            all_dfs.append(df)                  
        except:
          print('score.txt not found in {}'.format(log_path))

  return pd.concat(all_dfs)

def draw(data):
  env_titles = [
    'Sokoban-5x5-1',
    'Sokoban-6x6-1',
    'Sokoban-7x7-1',
    'Sokoban-6x6-3',
    'Sokoban-5x5-2',
    'Sokoban-6x6-2',
    'Sokoban-7x7-2',
  ]

  palette = [
      sns.color_palette('deep')[0],
      sns.color_palette('deep')[1],
      sns.color_palette('deep')[2],
      sns.color_palette('deep')[4],
      sns.color_palette('bright')[3],
  ]

  g = sns.FacetGrid(data, col="env", col_wrap=4, height=6, 
                    sharex=False, sharey=True, palette=palette, 
                    hue_order=['UER', 'PER', 'EBU', 'DisCor', 'TER'])

  g.map_dataframe(sns.lineplot, x="steps", y="normalized_mean", hue='algo',
      linewidth=6.0
  )
  g.add_legend(fontsize='30', title_fontsize='30', loc='lower right')

  leg_lines = g._legend.get_lines()
  plt.setp(leg_lines, linewidth=6)
  leg = g._legend
  leg.set_bbox_to_anchor([0.9, 0.1])  

  leg_lines = g._legend.get_lines()
  plt.setp(leg_lines, linewidth=6)
  g._legend.set_in_layout(True)
  g.fig.subplots_adjust(wspace=0.32)

  for idx, (ax, env_name) in enumerate(zip(g.axes.flat, env_titles)):
      labels = ax.get_xticklabels()
      ax.set_yticklabels(ax.get_yticks(), size=24)
      ax.set_xticklabels(ax.get_xticks(), size=24)
      ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: f'{x / 1e6:.1f}M')) 
      ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda y, p: f'{y:.2f}')) 
      xticks=ax.xaxis.get_major_ticks()
      for i in range(len(xticks)):
          if i % 2 == 1 and i < len(xticks) - 1:
              xticks[i].set_visible(False)
      ax.set_title(env_name, size=28)
      if idx % 4 == 0:               
          ax.set_ylabel('Normalized Return',fontsize=24)
      if idx // 4 == 1:
          ax.set_xlabel('Timestep', fontsize=24)
  g.fig.tight_layout()
  plt.savefig('figures/fig_overall_perf_sokoban.png')

def main():
  data = load_data()
  draw(data.reset_index())

if __name__ == '__main__':
  main()