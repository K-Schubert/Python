from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from btk import btk
import pandas as pd

import os

#Function that extract the data from the c3d files and built a table with predictors, contextual variable
#  and new labels for the response
def formatdata(filenam,filecomplete):
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(filecomplete)
    #reader.SetFilename("CP_GMFCS1_01916_20130128_18.c3d") 
    reader.Update()
    acq = reader.GetOutput()

    # get some parameters
    freq = acq.GetPointFrequency() # give the point frequency
    n_frames = acq.GetPointFrameNumber() # give the number of frames
    first_frame = acq.GetFirstFrame()

    # events
    n_events = acq.GetEventNumber()
    event = acq.GetEvent(1) # extract the first event of the aquisition
    #print(acq)
    label = event.GetLabel() # return a string representing the Label
    context = event.GetContext() # return a string representing the Context
    event_frame = event.GetFrame() # return the frame as an integer

    # get events
    n_events = acq.GetEventNumber()
    event_frames = pd.DataFrame([acq.GetEvent(event).GetFrame() for event in range(n_events)],columns=["Frame"])
    event_frames = pd.concat([event_frames,pd.DataFrame([acq.GetEvent(event).GetLabel() for event in range(n_events)],columns=["Label"])],axis=1)
    event_frames = pd.concat([event_frames,pd.DataFrame([acq.GetEvent(event).GetContext() for event in range(n_events)],columns=["Context"])],axis=1)
    event_frames = pd.concat([event_frames,pd.DataFrame([1]*n_events,columns=["Event"])],axis=1)
    event_frames = event_frames.sort_values(by=['Frame'])

    # metadata get the covariates with al the markers
    metadata = acq.GetMetaData()
    point_labels = metadata.FindChild("POINT").value().FindChild("LABELS").value().GetInfo().ToString()

    markers = list()
    start = False
    for label in point_labels:
        label = label.replace(' ', '')
        if label == 'C7':
            start = True
        if label == 'CentreOfMass':
            break
        if start:
            markers.append(label)
    markers
    # exemple on how to construct array with markers from one frame
    name=markers[0]
    Frame = 0
    data_FrameChosen=None
    data=None
    DF=None
    DFnew=pd.DataFrame()
    event_fr = [acq.GetEvent(event).GetFrame() for event in range(n_events)]
    event_fr.sort()
    #start_frame = first_frame
    end_frame = n_frames
    start_frame = event_fr[0]-first_frame
    #end_frame = event_fr[-1]-first_frame

    for frame in range(start_frame,end_frame):
        data_FrameChosen = np.array([acq.GetPoint(markers[0]).GetValues()[frame,:]])
        tot=np.array(data_FrameChosen)
        for name in markers[1:]:
            data_FrameChosen = np.array([acq.GetPoint(name).GetValues()[frame,:]])
            tot=np.append(tot,data_FrameChosen,axis=0)
        data=pd.DataFrame(tot)
        data=pd.DataFrame.transpose(data)
        data=data.set_index(pd.Index(['x1','x2','x3']))
        data.columns=markers
        DF=pd.DataFrame([frame]*3,index=["x1","x2","x3"],columns=['Frame'])
        DF=pd.concat([DF,data],axis=1,join='inner') 
        DFnew=pd.concat([DFnew,DF],join_axes=None)

    DFx1=DFnew.iloc[DFnew.index=="x1",:]
    DFx1=DFx1.set_index(pd.Index(DFx1.iloc[:,0]))
    DFx1.index.names=['index']
    DFx1.columns=np.append('Frame',[marker + "x1" for marker in markers])
    DFx2=DFnew.iloc[DFnew.index=="x2",:]
    DFx2=DFx2.set_index(pd.Index(DFx2.iloc[:,0]))
    DFx2.index.names=['index']
    DFx2.columns=np.append('Frame',[marker + "x2" for marker in markers])
    DFx3=DFnew.iloc[DFnew.index=="x3",:]
    DFx3=DFx3.set_index(pd.Index(DFx3.iloc[:,0]))
    DFx3.index.names=['index']
    DFx3.columns=np.append('Frame',[marker + "x3" for marker in markers])
    DFtot=pd.concat([DFx1,DFx2.iloc[:,1:],DFx3.iloc[:,1:]],axis=1)
    df4 = pd.merge(DFtot, event_frames, on='Frame',how='left')
    chara=filenam.split('_')

    k=0
    df4['Group']=pd.Series([chara[0]]*len(df4))
    if (chara[0]=="ITW"): k=1
    df4['Patho']=pd.Series([chara[1-k]]*len(df4))
    df4['Patient']=pd.Series([chara[2-k]]*len(df4))
    df4['Year']=pd.Series([chara[3-k][:4]]*len(df4))
    df4['Month']=pd.Series([chara[3-k][4:6]]*len(df4))
    df4['Day']=pd.Series([chara[3-k][6:]]*len(df4))
    df4['Trial']=pd.Series([chara[4-k][:chara[4-k].index('.')]]*len(df4))
    df4['Event']=df4['Event'].fillna(0)
    df4['Classes'] = pd.Series('na', index=df4.index)

    #Define the first frame strike or off, left or right
    if ((df4.loc[1,'Label']=='Foot Strike') & (df4.loc[1,'Context']=='Left')):
        df4.loc[0,'Classes']='R'
    #   df4.loc[0,'Classes2']='Right'
    if ((df4.loc[1,'Label']=='Foot Strike') & (df4.loc[1,'Context']=='Right')):
        df4.loc[0,'Classes']='L'
    #    df4.loc[0,'Classes2']='Left'
    if ((df4.loc[1,'Label']=='Foot Off') & (df4.loc[1,'Context']=='Left')):
        df4.loc[0,'Classes']='RL'
    #    df4.loc[0,'Classes2']='Right'
    if ((df4.loc[1,'Label']=='Foot Off') & (df4.loc[1,'Context']=='Right')):
        df4.loc[0,'Classes']='LR'
    #    df4.loc[0,'Classes2']='Left'

    for i in range(1,df4.shape[0]):
        if pd.notna(df4.loc[i,'Context']): 
            if ((df4.loc[i,'Label']=='Foot Strike') & (df4.loc[i,'Context']=='Left')):
                df4.loc[i,'Classes']='LR'
            #    df4.loc[i,'Classes2']=df4.loc[i-1,'Classes2']
            if ((df4.loc[i,'Label']=='Foot Strike') & (df4.loc[i,'Context']=='Right')):
                df4.loc[i,'Classes']='RL'
            #    df4.loc[i,'Classes2']=df4.loc[i-1,'Classes2']
            if ((df4.loc[i,'Label']=='Foot Off') & (df4.loc[i,'Context']=='Left')):
                df4.loc[i,'Classes']='R'
            #    df4.loc[i,'Classes2']=df4.loc[i-1,'Classes']
            if ((df4.loc[i,'Label']=='Foot Off') & (df4.loc[i,'Context']=='Right')):
                df4.loc[i,'Classes']='L'
            #    df4.loc[i,'Classes2']=df4.loc[i-1,'Classes']
        if pd.isna(df4.loc[i,'Context']):
            df4.loc[i,'Classes']=df4.loc[i-1,'Classes']
        #    df4.loc[i,'Classes2']=df4.loc[i-1,'Classes2']
    return(df4)


#Main, go through each file in the directory and concatenate the table created with the formatdata function
for j in ['CP','FD','ITW']:
    if (j =='CP'):
        path = '/Users/marc/Documents/DataMining/SOFAMEH/Sofamehack2019/Sub_DB_Checked/CP'
    if (j =='FD'):
        path = '/Users/marc/Documents/DataMining/SOFAMEH/Sofamehack2019/Sub_DB_Checked/FD'
    if (j =='ITW'):
        path = '/Users/marc/Documents/DataMining/SOFAMEH/Sofamehack2019/Sub_DB_Checked/ITW'
    files = []
    filesnames=[]
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.c3d' in file:
                files.append(os.path.join(r,file))
                filesnames.append(os.path.join(file))
    
    pdata=pd.DataFrame()
    pdataall=pd.DataFrame()
    for i in range(0,len(files)):
        filenam=filesnames[i]
        filecomplete=files[i]
        print(filenam)
        dfdata=formatdata(filenam,filecomplete)
        pdata=pd.concat([pdata,dfdata],axis=0,sort=False)
    if (j =='CP'):
        pdata.to_csv('/Users/marc/Documents/DataMining/SOFAMEH/Sofamehack2019/btk_mac_os/CP.csv',sep=',')
    if (j =='FD'):
        pdata.to_csv('/Users/marc/Documents/DataMining/SOFAMEH/Sofamehack2019/btk_mac_os/FD.csv',sep=',')
    if (j =='ITW'):
        pdata.to_csv('/Users/marc/Documents/DataMining/SOFAMEH/Sofamehack2019/btk_mac_os/ITW.csv',sep=',')
#print(pdata)