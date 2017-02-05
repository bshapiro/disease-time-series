config = {'gp_variance': None,  # no need to change
          'gp_lengthscale': None,  # no need to change
          'kernel': 'stack',  # no need to change
          'k': 70,  # set K
          'dir': 'clusters',  # no need to change
          'differential_transform': False,  # differential or 0-mean view
          'init': 'kmeans',  # kmeans or myeloma_paper
          'num_related': 2,  # set J
          'strength': 0.3,  # set strength parameter
          'dataset': 'ribosome',  # if single view, which view? ribosome or polya
          'parallel': True 
          }
