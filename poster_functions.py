import pandas as pd
import numpy as np # for numerical calculations such as histogramming
import matplotlib.pyplot as plt # for plotting
from matplotlib.lines import Line2D # for dashed line in legend
from matplotlib.ticker import AutoMinorLocator,MaxNLocator # for minor ticks and forcing integer tick labels
import matplotlib.image as mpimg
from sklearn.metrics import roc_curve, auc

# In[2]:

figy = 7
ATLAS_Open_Data_size = 15
for_education_size = 10
text_width_multiplier = 1.8
#plt.rcParams.update({'font.size': 12})

samples = {

    'data': {
        'list' : ['data_A','data_B','data_C','data_D']
    },
    
    r'$t\bar{t}Z$ signal' : {
        'list' : ['ttee','ttmumu'],
    },

    r'$t\bar{t}$' : {
        'list' : ['ttbar_lep'],
    },
    
    'High pT Z' : {
        'list' : ['Zmumu_PTV500_1000','Zmumu_PTV1000_E_CMS','Zee_PTV500_1000','Zee_PTV1000_E_CMS'],
    },
    
    'Z + b' : {                                                                                                                                                                                  
        'list' : ['Zmumu_PTV0_70_BFilter','Zmumu_PTV70_140_BFilter','Zmumu_PTV140_280_BFilter',
                  'Zmumu_PTV280_500_BFilter','Zee_PTV0_70_BFilter','Zee_PTV70_140_BFilter',
                  'Zee_PTV140_280_BFilter','Zee_PTV280_500_BFilter'], 
        # 'Ztautau_PTV0_70_BFilter','Ztautau_PTV70_140_BFilter','Ztautau_PTV140_280_BFilter' 
        # return 0 events in 2b6j Signal Region                                                                                                                                                                   
    },
    
    'Z + c' : {                                                                                                                                                                                  
        'list' : ['Zmumu_PTV140_280_CFilterBVeto','Zmumu_PTV280_500_CFilterBVeto',
                  'Zee_PTV140_280_CFilterBVeto','Zee_PTV280_500_CFilterBVeto',
                 'Zmumu_PTV70_140_CFilterBVeto','Zee_PTV70_140_CFilterBVeto'], 
        # 'Zmumu_PTV70_140_CFilterBVeto',
        # 'Zee_PTV70_140_CFilterBVeto',
        # return low yields in 2b6j Signal Region
        # 'Ztautau_PTV0_70_CFilterBVeto','Ztautau_PTV70_140_CFilterBVeto','Ztautau_PTV140_280_CFilterBVeto',
        # 'Ztautau_PTV280_500_CFilterBVeto','Zmumu_PTV0_70_CFilterBVeto','Zee_PTV0_70_CFilterBVeto'
        # return 0 events in 2b6j Signal Region
    },
    
    'Z + light' : {                                                                                                                                                                              
        'list' : ['Zmumu_PTV140_280_CVetoBVeto','Zmumu_PTV280_500_CVetoBVeto','Zee_PTV140_280_CVetoBVeto',
                  'Zee_PTV280_500_CVetoBVeto'], 
        # 'Zmumu_PTV140_280_CVetoBVeto',
        # 'Zmumu_PTV280_500_CVetoBVeto',
        # 'Zee_PTV140_280_CVetoBVeto',
        # 'Zee_PTV280_500_CVetoBVeto'
        # return low yields in 2b6j Signal Region
        # 'Zmumu_PTV70_140_CVetoBVeto','Ztautau_PTV0_70_CVetoBVeto','Ztautau_PTV70_140_CVetoBVeto',
        # 'Ztautau_PTV140_280_CVetoBVeto','Ztautau_PTV280_500_CVetoBVeto','Zee_PTV0_70_CVetoBVeto',
        # 'Zmumu_PTV0_70_CVetoBVeto','Zee_PTV70_140_CVetoBVeto' return 0 events in 2b6j Signal Region     
    },                                                                                                                                                                                                                                                                                                                                                                                      

    'Other' : {                                                                                                                                                                                  
        'list' : ['ttW',
                  'Ztautau_PTV280_500_BFilter','Ztautau_PTV500_1000','Ztautau_PTV1000_E_CMS','llll','lllv','llvv'], 
        # 'Ztautau_PTV280_500_BFilter',
        # 'Ztautau_PTV500_1000',
        # 'Ztautau_PTV1000_E_CMS'
        # 'ZH125_ZZ4lep'
        # 'VBFH125_ZZ4lep',
        # 'WH125_ZZ4lep',
        # 'ggH125_ZZ4lep',
        # 'llll',
        # 'lllv',
        # 'llvv',
        # 'single_top_wtchan',
        # 'ttW',
        # return low yields in 2b6j Signal Region
        # 'ggH125_tautaull','VBFH125_WW2lep','ggH125_WW2lep','WpH125J_qqWW2lep','ZH125J_vvWW2lep','VBFH125_tautaull',
        # 'ZH125J_qqWW2lep' return 0 events for 2b6j Signal Region 
    },
    
}


# In[4]:


data_dict = {} # define empty dictionary that will hold all Data and MC events passing the 2b6j SR requirements
for s in samples: # loop over the file groupings in the 'samples' dictionary defined above
    frames = [] # define empty list that will hold the dataframes for this file grouping
    for val in samples[s]['list']: # loop over the individual files within this file grouping
        temp_DataFrame = pd.read_csv('results_ttZ_2l_2b6j_cut_lep_type_SR/dataframe_id_'+val+'.csv',index_col='entry')
        frames.append(temp_DataFrame) # append the dataframe for this individual file to the list of dataframes
    data_dict[s] = pd.concat(frames) # concatenate the dataframes for this file grouping into one


# In[5]:


frames_0 = [] # define empty list to hold the dataframes for files with 0 heavy-flavour partons
frames_1 = [] # define empty list to hold the dataframes for files with 1 heavy-flavour parton
frames_2 = [] # define empty list to hold the dataframes for files with 2 or more heavy-flavour partons
for s in ['High pT Z', 'Z + b', 'Z + c', 'Z + light']: # loop over the Z+jets file groupings
    temp_0 = data_dict[s][data_dict[s]['HF_n']==0] # find the events with 0 heavy-flavour partons
    temp_1 = data_dict[s][data_dict[s]['HF_n']==1] # find the events with 1 heavy-flavour parton
    temp_2 = data_dict[s][data_dict[s]['HF_n']>=2] # find the events with 2 heavy-flavour partons
    frames_0.append(temp_0) # append the dataframe containing 0 heavy-flavour partons
    frames_1.append(temp_1) # append the dataframe containing 1 heavy-flavour parton
    frames_2.append(temp_2) # append the dataframe containing 2 heavy-flavour partons
data_dict['Z + 0 HF'] = pd.concat(frames_0) # concatenate the dataframes containing 0 heavy-flavour partons into one
data_dict['Z + 1 HF'] = pd.concat(frames_1) # concatenate the dataframes containing 1 heavy-flavour parton into one
data_dict['Z + 2 HF'] = pd.concat(frames_2) # concatenate the dataframes containing 2 heavy-flavour partons into one
del data_dict['High pT Z'] # delete the 'High pT Z' entry in the dictionary of dataframes
del data_dict['Z + b'] # delete the 'Z + b' entry in the dictionary of dataframes
del data_dict['Z + c'] # delete the 'Z + c' entry in the dictionary of dataframes
del data_dict['Z + light'] # delete the 'Z + light' entry in the dictionary of dataframes


# In[6]:


samples_HF = {

    'data': {
        'list' : ['data_A','data_B','data_C','data_D']
    },
    
    'Other' : {
        'color' : "#79b278" # purple                                                                                                                                                                   
    },  
    
    'Z + 0 HF' : {
        'color' : "#ce0000" # light green                                                                                          #                                                                    
    },
    
    'Z + 1 HF' : {
        'color' : "#ffcccc" # middle green
    }, 
    
    'Z + 2 HF' : {
        'color' : "#ff6666" # dark green
    }, 
    
    r'$t\bar{t}$' : {
        'color' : "#f8f8f8" # almost white
    }, 
    
    r'$t\bar{t}Z$ signal' : {
        'color' : "#00ccfd" # blue
    },                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    
}


ptll = {
    # change plotting parameters  
    'xlabel':r'$\mathrm{p_{T}^{ll}}$ [GeV]', # x-axis label
}

Nmbjj_top = {
    # change plotting parameters
    'xlabel':r'$N_{bjj}^{top-mass}$', # x-axis label
}

H1 = {
    # change plotting parameters 
    'xlabel':'H1', # x-axis label
}

CentrJet = {
    # change plotting parameters
    'xlabel':r'$Centr_{jet}$', # x-axis label
}

eta_ll = {
    # change plotting parameters
    'xlabel':r'$\eta_{ll}$', # x-axis label
}

NJetPairsZMass = {
    # change plotting parameters
    'xlabel':r'$N^{Vmass}_{jj}$', # x-axis label
}

dRjjave_jet = {
    # change plotting parameters
    'xlabel':r'$\Delta R_{jj}^{ave}$', # x-axis label
}

pt4_jet = {
    # change plotting parameters
    'xlabel':r'$\mathrm{p_{T}^{4jet}}$ [GeV]', # x-axis label
}

dRbb = {
    # change plotting parameters
    'xlabel':r'$\Delta R_{bb}$', # x-axis label
}

MbbPtOrd = {
    # change plotting parameters   
    'xlabel':r'$\mathrm{M_{bb}^{pTord}}$ [GeV]', # x-axis label
}

HT_jet6 = {
    # change plotting parameters
    'xlabel':r'$\mathrm{H_{T}^{6jets}}$ [GeV]', # x-axis label
}

pt6_jet = {
    # change plotting parameters
    'xlabel':r'$\mathrm{p_{T}^{6jet}}$ [GeV]', # x-axis label
}

dRll = {
    # change plotting parameters
    'xlabel':r'$\Delta R_{ll}$', # x-axis label
}

MaxMMindRlepb = {
    # change plotting parameters  
    'xlabel':r'Max$\mathrm{M^{mindR}_{lepb}}$ [GeV]', # x-axis label
}

MjjMindR = {
    # change plotting parameters 
    'xlabel':r'$\mathrm{M^{mindR}_{jj}}$ [GeV]', # x-axis label
}

MWavg = {
    # change plotting parameters
    'xlabel':r'$\mathrm{M^{avg}_{W}}$ [GeV]', # x-axis label
}

pT1b = {
    # change plotting parameters
    'xlabel':r'$\mathrm{p_{T}^{b1}}$ [GeV]', # x-axis label
}

hist_dict = {'ptll':ptll,
             'Nmbjj_top':Nmbjj_top,'H1':H1,
             'CentrJet':CentrJet,'eta_ll':eta_ll,'NJetPairsZMass':NJetPairsZMass,
             'dRjjave_jet':dRjjave_jet,'pt4_jet':pt4_jet,'dRbb':dRbb,'MbbPtOrd':MbbPtOrd,
             'HT_jet6':HT_jet6,'pt6_jet':pt6_jet,'dRll':dRll,
             'MaxMMindRlepb':MaxMMindRlepb,'MjjMindR':MjjMindR,'MWavg':MWavg,
             'pT1b':pT1b}
            # add a histogram here if you want it plotted



def plot_separation():

    signal = r'$t\bar{t}Z$ signal' # label for signal
    
    # *******************
    # general definitions (shouldn't need to change)
    
    for x_variable,hist in MVA_hist_dict.items(): # loop variables

        h_xrange_min = hist['xrange_min'] # get the x-axis minimum defined in hist_dict
        h_xrange_max = hist['xrange_max'] # get the x-axis maximum defined in hist_dict
        h_bin_width = hist['bin_width'] # get the bin width defined in hist_dict
        h_xlabel = hist['xlabel'] # get the x-axis label defined in hist_dict
        h_title = hist['title'] # get the histogram title defined in hist_dict
        h_linear_top_margin = hist['linear_top_margin'] # to decrease the separation between data and the top of the figure, pick a number closer to 1
    
        bin_edges = np.arange(start=h_xrange_min, # The interval includes this value
                     stop=h_xrange_max+h_bin_width, # The interval doesn't include this value
                     step=h_bin_width ) # Spacing between values
        bin_centres = np.arange(start=h_xrange_min+h_bin_width/2, # The interval includes this value
                            stop=h_xrange_max+h_bin_width/2, # The interval doesn't include this value
                            step=h_bin_width ) # Spacing between values
        
        # clip signal underflow and overflow into x-axis range
        signal_x = np.clip(data_dict[signal][x_variable].values,
                           h_xrange_min, h_xrange_max )
        
        # signal entry weights
        signal_weights = data_dict[signal].totalWeight.values 
        
        mc_x = [] # define list to hold the MC histogram entries
        mc_weights = [] # define list to hold the MC weights

        for s in samples_HF: # loop over samples
            if s!='data' and s!=signal: # if not data nor signal
                mc_x = [*mc_x, # mc_x for previous sample
                        *np.clip(data_dict[s][x_variable].values,
                                 h_xrange_min,h_xrange_max) ] # this sample
                mc_weights = [*mc_weights, # weights for previous sample 
                              *data_dict[s].totalWeight.values ] # this sample
                
    
        #fig = plt.figure() # define new figure to plot
        fig, axs = plt.subplots(1, 3, figsize=(3*9/8*figy*4/3, figy))

        # *************
        # Main plot 
        # *************
        #fig.add_subplot(121) # add subplot for separation
        #plt.clf() # clear figure
        
        # plot the background Monte Carlo distribution
        mc_heights = axs[0].hist(mc_x, bins=bin_edges, 
                                    density=True, # area under histogram will sum to 1
                                    weights=mc_weights, 
                                    histtype='step', color='red', 
                                    label='Total background' )
        
        
        # get un-normaliased background Monte Carlo entries
        mc_heights_not_normalised,_ = np.histogram(mc_x, 
                                                   bins=bin_edges,
                                                   weights=mc_weights )
        
        
        
        # plot the signal distribution
        signal_heights = axs[0].hist(signal_x, bins=bin_edges, 
                                    density=True, # area under histogram will sum to 1
                                    weights=signal_weights, 
                                    histtype='step', color=samples_HF[signal]['color'], 
                                    label=signal, 
                                        linestyle='--' ) # dashed line
        
        # get the un-normalised signal entries
        signal_heights_not_normalised,_ = np.histogram(signal_x,
                                                       bins=bin_edges,
                                                       weights=signal_weights )
        
        
        
        separation = 0 # start separation counter at 0
        nstep  = int((h_xrange_max-h_xrange_min)/h_bin_width) # number of bins
        nS     = sum(signal_heights_not_normalised)*h_bin_width # signal integral
        nB     = sum(mc_heights_not_normalised)*h_bin_width # background integral
        for bin_i in range(nstep): # loop over each bin
            # normalised signal in bin_i
            s = signal_heights_not_normalised[bin_i]/nS
            # normalised background in bin_i
            b = mc_heights_not_normalised[bin_i]/nB
            # separation
            if (s + b > 0): separation += (s - b)*(s - b)/(s + b)
        separation *= (0.5*h_bin_width) # multiply by 0.5 x bin_width
        
        

        # x-axis label
        axs[0].set_xlabel(h_xlabel)
                                                           
        # write y-axis label for main axes
        axs[0].set_ylabel('Arbitrary units') 

        # set y-axis limits for main axes
        axs[0].set_ylim(bottom=0, 
                           top=axs[0].get_ylim()[1]*h_linear_top_margin )
        

        # Add text 'ATLAS Open Data' on plot
        plt.text(0.05, # x
                 0.91, # y
                 'ATLAS Open Data', # text
                 transform=axs[0].transAxes, # coordinate system used is that of main_axes
                 fontsize=ATLAS_Open_Data_size ) 

        # Add text 'for education' on plot
        plt.text(0.05, # x
                 0.86, # y
                 'for education', # text
                 transform=axs[0].transAxes, # coordinate system used is that of main_axes
                 style='italic',
                 fontsize=for_education_size ) 
        
        # print separation value on plot
        plt.text(0.05, # x
                 0.80, # y
                 'Separation: '+str(round(separation*100,3))+'%',
                 transform=axs[0].transAxes ) # coordinate system used is that of main_axes
    
        # Create new legend handles but use existing colors
        handles, labels = axs[0].get_legend_handles_labels()
        handles[labels.index(signal)] = Line2D([], [], c=samples_HF[signal]['color'], 
                                               linestyle='dashed' )
        handles[labels.index('Total background')] = Line2D([], [], 
                                                           c='red' )
        
        # draw the legend
        axs[0].legend(handles=handles, labels=labels) # no box around the legend
        
        axs[0].set_title('Separation')
        
    if hasattr(bdt, "decision_function"): # if BDT                                                  
        train_decisions = bdt.decision_function(X_train) # get decisions on train set               
        test_decisions = bdt.decision_function(X_test) # get decisions on test set                  

    # Compute ROC curve for training set                                                            
    train_fpr, train_tpr, _ = roc_curve(y_train, # actual                                           
                                        train_decisions ) # predicted                               

    # Compute area under the curve for training set                                                 
    train_roc_auc = auc(train_fpr, # false positive rate                                            
                        train_tpr ) # true positive rate                                            

    #fig.add_subplot(122) # add subplot for roc
    #ax1 = plt.gca()
    #ax1.set_position([0.7,0,1,0.75]) # [left, bottom, width, height]
    
    # plot train ROC curve                                                                          
    axs[1].plot(train_tpr, # x                                                                         
             1-train_fpr, # y                                                                       
             label='$A_{train}$ = %0.2f'%(train_roc_auc), color='red' )

    # Compute ROC curve for test set                                                                
    test_fpr, test_tpr, _ = roc_curve(y_test, # actual                                              
                                      test_decisions ) # predicted                                  

    # Compute aread under the curve for test set                                                    
    test_roc_auc = auc(test_fpr, test_tpr)

    # plot test ROC curve                                                                           
    axs[1].plot(test_tpr, # x                                                                          
             1-test_fpr, # y                                                                        
             label='$A_{test}$ = %0.2f'%(test_roc_auc), color='blue' )

    # Add text 'ATLAS Open Data' on plot                                                            
    plt.text(0.05, # x                                                                              
             0.7, # y                                                                              
             'ATLAS Open Data', # text                                                              
             transform=axs[1].transAxes, # coordinate system used is that of ax1                       
             fontsize=ATLAS_Open_Data_size )

    # Add text 'for education' on plot                                                              
    plt.text(0.05, # x                                                                              
             0.65, # y                                                                              
             'for education', # text                                                                
             transform=axs[1].transAxes, # coordinate system used is that of ax1                       
             style='italic',
             fontsize=for_education_size )

    # Add text indicating colour of training sample                                                 
    plt.text(0.05, # x
             0.57, # y
         'training sample', # text
             color='red',
         transform=axs[1].transAxes ) # coordinate system used is that of ax1

    # Add text indicating colour of testing sample
    plt.text(0.05, # x
             0.52, # y
             'testing sample', # text
             color='blue',
         transform=axs[1].transAxes ) # coordinate system used is that of ax1

    axs[1].set_xlabel( 'signal efficiency' ) # add x-axis label
    axs[1].set_ylabel('background rejection') # add y-axis label
    axs[1].set_title('ROC curve') # add title
    axs[1].legend() # draw legend
        
    img = mpimg.imread('ROC_text.png')
    ll, bb, ww, hh = axs[2].get_position().bounds
    axs[2].set_position([ll-0.04,bb-0.3,ww*text_width_multiplier,hh*text_width_multiplier]) # [left, bottom, width, height]
    axs[2].imshow(img)
    axs[2].axis('off') 


# In[42]:


def plot_data(scaling_factor=1,save_postfix='',region_label='2l-2b6j-SR'):
    
    # *******************
    # general definitions (shouldn't need to change)
      
    for x_variable,hist in MVA_hist_dict.items(): # loop variables

        h_xrange_min = hist['xrange_min'] # get the x-axis minimum defined in hist_dict
        h_xrange_max = hist['xrange_max'] # get the x-axis maximum defined in hist_dict
        h_bin_width = hist['bin_width'] # get the bin width defined in hist_dict
        h_xlabel = hist['xlabel'] # get the x-axis label defined in hist_dict
        h_title = hist['title'] # get the histogram title defined in hist_dict
        h_linear_top_margin = hist['linear_top_margin'] # to decrease the separation between data and the top of the figure, pick a number closer to 1
    
        bin_edges = np.arange(start=h_xrange_min, # The interval includes this value
                     stop=h_xrange_max+h_bin_width, # The interval doesn't include this value
                     step=h_bin_width ) # Spacing between values
        bin_centres = np.arange(start=h_xrange_min+h_bin_width/2, # The interval includes this value
                            stop=h_xrange_max+h_bin_width/2, # The interval doesn't include this value
                            step=h_bin_width ) # Spacing between values
    
        mc_x = [] # define list to hold the MC histogram entries
        mc_weights = [] # define list to hold the MC weights
        mc_colors = [] # define list to hold the MC bar colors
        mc_labels = [] # define list to hold the MC legend labels

        mc_stat_err_squared = np.zeros(len(bin_centres)) # define array to hold the MC statistical uncertainties
        for s in samples_HF: # loop over samples
            if s!='data': # if not data
                # clip the underflow and overflow into the x-axis range
                clipped_x_variable = np.clip(data_dict[s][x_variable], h_xrange_min, h_xrange_max)
                mc_x.append( clipped_x_variable ) # append to the list of Monte Carlo histogram entries
                totalWeights = data_dict[s]['totalWeight'] # get the totalWeight column
                mc_weights.append( totalWeights ) # append to the list of Monte Carlo weights
                mc_colors.append( samples_HF[s]['color'] ) # append to the list of Monte Carlo bar colors
                mc_labels.append( s ) # append to the list of Monte Carlo legend labels 
                weights_squared,_ = np.histogram(clipped_x_variable, bins=bin_edges,
                                                 weights=totalWeights**2) # square the totalWeights
                mc_stat_err_squared = np.add(mc_stat_err_squared,weights_squared) # add weights_squared for s 
    
    
        fig, axs = plt.subplots(1, 2, figsize=(2*9/8*figy*4/3, figy))        
 
        # plot the Monte Carlo bars
        mc_heights = axs[0].hist(mc_x, bins=bin_edges, 
                                    weights=mc_weights, stacked=True, 
                                    color=mc_colors, label=mc_labels )

        mc_x_tot = mc_heights[0][-1] # stacked background MC y-axis value
        mc_x_err = np.sqrt( mc_stat_err_squared ) # statistical error on the MC bars
        
        # histogram the data
        data_x,_ = np.histogram(np.clip(data_dict['data'][x_variable].values,
                                        h_xrange_min,h_xrange_max), 
                                bins=bin_edges ) 
        data_x_no0_in_data = [] # empty list to hold data bins with non-0 entries
        bin_centres_no0_in_data = [] # empty list to hold bin_centres with non-0 entries
        mc_x_tot_no0_in_data = [] # empty list to hold MC bins with non-0 data entries
        for i in range(len(data_x)): # loop over bins
            if data_x[i]!=0: # non-0 entry in bin i
                data_x_no0_in_data.append(data_x[i]) # append to list of data bins with non-0 entries
                bin_centres_no0_in_data.append(bin_centres[i]) # append to list of bin_centres with non-0 data entries
                mc_x_tot_no0_in_data.append(mc_x_tot[i]) # append to list of MC bins with non-0 data entries
        
        # statistical error on the data
        data_x_errors = np.sqrt(data_x_no0_in_data)
        
        # plot the data points
        axs[0].errorbar(x=bin_centres_no0_in_data, 
                           y=scaling_factor*np.array(data_x_no0_in_data), 
                           yerr=np.sqrt(scaling_factor)*data_x_errors,
                           fmt='ko', # 'k' means black and 'o' is for circles 
                           label='Data')
        
        # plot the statistical uncertainty
        axs[0].bar(bin_centres, # x
                  2*mc_x_err, # heights
                  alpha=0.5, # half transparency
                  bottom=mc_x_tot-mc_x_err, color='none', 
                  hatch="////", width=h_bin_width, label='Stat. Unc.' )
        
        # set the x-limit of the main axes
        axs[0].set_xlim( left=h_xrange_min, right=h_xrange_max ) 

        # separation of x axis minor ticks
        axs[0].xaxis.set_minor_locator( AutoMinorLocator() ) 

        # set the axis tick parameters for the main axes
        axs[0].tick_params(which='both', # ticks on both x and y axes
                              direction='in', # Put ticks inside and outside the axes
                              top=True, # draw ticks on the top axis
                              right=True ) # draw ticks on right axis
        
        # x-axis label
        axs[0].set_xlabel(h_xlabel, x=1, 
                             horizontalalignment='right' )
        
        if len(h_xlabel.split('['))>1: # if x-axis has units
            # find x-axis units
            y_units = ' '+h_xlabel[h_xlabel.find("[")+1:h_xlabel.find("]")]
        else: y_units = '' # no x-axis units
            
        # y-axis label
        axs[0].set_ylabel(r'Events / '+str(round(h_bin_width,3))+y_units,
                             y=1, horizontalalignment='right' )
        
        # force y-axis ticks to be integers
        axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # set y-axis limits for main axes
        axs[0].set_ylim(bottom=0,
                           top=scaling_factor*(np.amax(data_x)+np.sqrt(np.amax(data_x)))*h_linear_top_margin )
        
        # add minor ticks on y-axis for main axes
        axs[0].yaxis.set_minor_locator( AutoMinorLocator() ) 

        # Add text 'ATLAS Open Data' on plot
        plt.text(0.05, # x
                 0.91, # y
                 'ATLAS Open Data', # text
                 transform=axs[0].transAxes, # coordinate system used is that of main_axes
                 fontsize=ATLAS_Open_Data_size ) 

        # Add text 'for education' on plot
        plt.text(0.05, # x
                 0.86, # y
                 'for education', # text
                 transform=axs[0].transAxes, # coordinate system used is that of main_axes
                 style='italic',
                 fontsize=for_education_size ) 

        # Add energy and luminosity
        plt.text(0.05, # x
                 0.80, # y
                 '$\sqrt{s}$=13 TeV, 10 fb$^{-1}$', # text
                 transform=axs[0].transAxes ) # coordinate system used is that of main_axes
        
        # Add a label for the analysis region
        plt.text(0.05, # x
             0.74, # y
             '2L-Z-2b6j', # text 
             transform=axs[0].transAxes ) # coordinate system used is that of main_axes
        
        # draw the legend
        axs[0].legend(ncol=2, # number of columns
                         frameon=False ) # no box around the legend 
        
        axs[0].set_title('BDT distribution')
     
    img = mpimg.imread('data_text.png')
    ll, bb, ww, hh = axs[1].get_position().bounds
    axs[1].set_position([ll,bb-hh*text_width_multiplier+hh,ww*text_width_multiplier*4/3,hh*text_width_multiplier*4/3]) # [left, bottom, width, height]
    axs[1].imshow(img)
    axs[1].axis('off')

# In[10]:


BDT_inputs = ['dRjjave_jet','dRbb','HT_jet6','dRll','MWavg','pT1b',
              'ptll','NJetPairsZMass','pt4_jet','Nmbjj_top',
              'MbbPtOrd','MjjMindR','eta_ll','pt6_jet','CentrJet',
              'H1','MaxMMindRlepb'] # variables to use in BDT


# In[11]:


# for sklearn data is usually organised                                                                                                                                           
# into one 2D array of shape (n_samples x n_features)                                                                                                                             
# containing all the data and one array of categories                                                                                                                             
# of length n_samples  

all_MC = [] # define empty list that will contain all features for the MC
for key in samples_HF: # loop over the different keys in the dictionary of dataframes
    if key!='data': # only MC should pass this
        all_MC.append(data_dict[key][BDT_inputs]) # append the MC dataframe to the list containing all MC features
X = np.concatenate(all_MC) # concatenate the list of MC dataframes into a single 2D array of features, called X

all_y = [] # define empty list that will contain labels whether an event in signal or background
for key in samples_HF: # loop over the different keys in the dictionary of dataframes
    if key!=r'$t\bar{t}Z$ signal' and key!='data': # only background MC should pass this
        all_y.append(np.zeros(data_dict[key].shape[0])) # background events are labelled with 0
    elif key==r'$t\bar{t}Z$ signal':
        all_y.append(np.ones(data_dict[r'$t\bar{t}Z$ signal'].shape[0])) # signal events are labelled with 1
y = np.concatenate(all_y) # concatenate the list of lables into a single 1D array of labels, called y



from sklearn.model_selection import train_test_split

# make train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                  #test_size=0.33, 
                                                  random_state=492 )


# In[15]:



from sklearn.ensemble import GradientBoostingClassifier # BoostType

bdt = GradientBoostingClassifier(#n_estimators=500, # NTrees
                                 #max_depth=3,
                                #learning_rate=0.3, # Shrinkage
                                #min_samples_leaf=math.ceil(0.05*len(X_train)), # Minnodesize
                                #min_impurity_decrease=0.5
) # NodePurityLimit
bdt.fit(X_train, y_train)#, sample_weight=sample_weight) # fit BDT to training set


def correlations_and_overtraining(bins=20, **kwds):

    corrmat = all_MC[-1].corr(**kwds)

    fig, axs = plt.subplots(1, 3, figsize=(3*9/8*figy*4/3, figy))

    opts = {'cmap': plt.get_cmap("rainbow"), # use rainbow colormap
            'vmin': -1, # set colorbar minimum to -1
            'vmax': +1} # set colorbar maximum to 1
    heatmap1 = axs[0].pcolor(corrmat, **opts) # get heatmap
    plt.colorbar(heatmap1, ax=axs[0]) # plot colorbar

    axs[0].set_title("Correlations") # set title


    x_variables = corrmat.columns.values # get variables from data columns
    labels = [] # list to hold axis labels
    for x_variable in x_variables: # loop over variables
        label_no_units = hist_dict[x_variable]['xlabel'] # get label from histogram dictionary
        if '[' in label_no_units: label_no_units = label_no_units.split(' [')[0] # remove units
        labels.append(label_no_units) # append to list of labels
    # shift location of ticks to center of the bins
    axs[0].set_xticks(np.arange(len(labels))+0.5) # y-tick for each label
    axs[0].set_yticks(np.arange(len(labels))+0.5) # x-tick for each label
    axs[0].set_xticklabels(labels, rotation=85) # rotate by 85 degrees
    axs[0].set_yticklabels(labels) # horizontal

    decisions = [] # list to hold decisions of classifier
    for X,y in ((X_train, y_train), (X_test, y_test)): # train and test
        if hasattr(bdt, "decision_function"): # if BDT
            d1 = bdt.decision_function(X[y>0.5]).ravel() # signal
            d2 = bdt.decision_function(X[y<0.5]).ravel() # background
        decisions += [d1, d2] # add to list of classifier decision
        
    low = min(np.min(d) for d in decisions) # get minimum score
    high = max(np.max(d) for d in decisions) # get maximum score
    
    # map classifier decision to range [-1,1]
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    decisions_mapped = ((decisions-low)/(high-low))*2 - 1
        
    low = -1 # minimum mapped classifier score
    high = 1 # maximum mapped classifier score
    low_high = (low,high) # tuple holding mapped score range
    
    
    # plot the test set background mapped to the range [-1,1]
    background_test_heights = axs[1].hist(decisions_mapped[3], # background in test set
             bins=bins, # use number of bins in function definition
             range=low_high, # lower and upper range of the bins
             density=True, # area under the histogram will sum to 1
             histtype='stepfilled', # lineplot that's filled
             color='red', label='Background (test)', # Background (test)
             alpha=0.5 ) # half transparency
    
    # plot the test set signal mapped to the range [-1,1]
    signal_test_heights = axs[1].hist(decisions_mapped[2], # signal in test set
             bins=bins, # number of bins in function definition
             range=low_high, # lower and upper range of the bins
             density=True, # area under the histogram will sum to 1
             histtype='stepfilled', # lineplot that's filled
             color=samples_HF[r'$t\bar{t}Z$ signal']['color'], label='Signal (test)',
             alpha=0.5 ) # half transparency
    
    # histogram the training set background
    background_train_hist, bin_edges = np.histogram(decisions_mapped[1], # training background
                                   bins=bins, # number of bins in function definition
                                   range=low_high, # lower & upper range of the bins
                                   density=True ) # area under the histogram will sum to 1
    
    # get scale between raw and normalised training background
    background_scale = len(decisions[1]) / sum(background_train_hist) 
    
    # get statistical error on background test set
    background_test_err = np.sqrt(background_test_heights[0] * background_scale) / background_scale
    
    # get statistical error on background training set
    background_train_err = np.sqrt(background_train_hist * background_scale) / background_scale
    
    width = (bin_edges[1] - bin_edges[0]) # histogram bin width
    center = (bin_edges[:-1] + bin_edges[1:]) / 2 # bin centres
    
    # plot training set background
    axs[1].errorbar(x=center, y=background_train_hist, 
                 yerr=background_train_err,
                 fmt='ro', # red circles
                 label='Background (train)' ) # Background (train)

    # histogram the training set signal
    signal_train_hist, bin_edges = np.histogram(decisions_mapped[0], # training signal
                                   bins=bins, # number of bins in function definition
                                   range=low_high, # lower & upper range of the bins
                                   density=True ) # area under the histogram will sum to 1
    
    # get scale between raw and normalised training signal
    signal_scale = len(decisions[1]) / sum(signal_train_hist) 
    
    # get statistical error on signal test set
    signal_test_err = np.sqrt(signal_test_heights[0] * signal_scale) / signal_scale
    
    # get statistical error on signal training set
    signal_train_err = np.sqrt(signal_train_hist * signal_scale) / signal_scale
    
    # plot training set signal
    axs[1].errorbar(x=center, y=signal_train_hist, 
                 yerr=signal_train_err,
                       color=samples_HF[r'$t\bar{t}Z$ signal']['color'],
                 fmt='o', # circles
                 label='Signal (train)' ) # Signal (train)

    # write y-axis label
    axs[1].set_ylabel('Arbitrary units')
    
    axs[1].set_xlabel('BDT output')
    
    # Add text 'ATLAS Open Data' on plot
    plt.text(0.05, # x
             0.92, # y
             'ATLAS Open Data', # text
             transform=axs[1].transAxes, # coordinate system used is that of ax1
             fontsize=ATLAS_Open_Data_size ) 

    # Add text 'for education' on plot
    plt.text(0.05, # x
             0.87, # y
             'for education', # text
             transform=axs[1].transAxes, # coordinate system used is that of ax1
             style='italic',
             fontsize=for_education_size )
    
    axs[1].legend(loc='center left') # draw legend
    axs[1].set_ylim( bottom=0 ) # set y-axis limit
    
    # add plot title
    axs[1].set_title('Overtraining check')

    img = mpimg.imread('overtraining_text.png')
    ll, bb, ww, hh = axs[2].get_position().bounds
    axs[2].set_position([ll-0.04,bb-0.3,ww*text_width_multiplier,hh*text_width_multiplier]) # [left, bottom, width, height]
    axs[2].imshow(img)
    axs[2].axis('off')

# In[21]:


all_data_MC = [] # list to hold all data and MC
for key in samples_HF: # loop over samples
    all_data_MC.append(data_dict[key][BDT_inputs]) # get BDT_inputs from DataFrames
X_data_MC = np.concatenate(all_data_MC) # data & MC in one dataframe

# predict whether every event is signal or background
y_predicted = bdt.decision_function(X_data_MC)  

# map the BDT output to the range [-1,1]
y_predicted_mapped = ((y_predicted-min(y_predicted))/(max(y_predicted)-min(y_predicted)))*2 - 1


# In[22]:


cumulative_events = 0
for key in samples_HF:
    data_dict[key]['BDT_te'] = y_predicted_mapped[cumulative_events:cumulative_events+len(data_dict[key])]
    cumulative_events += len(data_dict[key])


# In[23]:


BDT_te = {
    # change plotting parameters
    'xrange_min':-1, # x-axis minimum
    'xrange_max':1, # x-axis maximum
    'bin_width':0.1, # width of each histogram bin
    'xlabel':'BDT output', # x-axis label
    'title':'BDT output', # histogram title

    # change aesthetic parameters if you want                                                                                                                                                                                                                    
    'linear_top_margin':1.1 # to decrease the separation between data and the top of the figure, pick a number closer to 1
}

MVA_hist_dict = {'BDT_te':BDT_te}





