# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 15:45:32 2016

@author: jchazin
"""

import numpy as np

data_names = ('0','2','5','7','8','9','11','14','16','17',
              '18','20','23','25','26','27','28','29','30',
              '31','32','33','34','35','36','37','38','39',
              '40','41','42','43','44','45','46','47','48',
              '49','50','51','52','53','54','55','56','57',
              '58','59','60','62','63','64','label')
              

dtype={'names': ('0', '2', '5', '7','8', '9', '11', '14',
                 '16', '17', '18', '20', '23', '25',
                 '26', '27', '28', '29', '30', '31',
                 '32', '33', '34', '35', '36',
                 '37', '38', '39', '40', '41', '42',
                 '43', '44', '45', '46', '47', '48',
                 '49', '50','51', '52', '53', '54',
                 '55', '56', '57', '58', '59', 
                 '60', '62', '63', '64','label'),
       'formats': ('S1', 'i4', 'f4')}


data = np.genfromtxt('data.csv',delimiter = ',',names=data_names,skip_header=1,dtype=None,)


