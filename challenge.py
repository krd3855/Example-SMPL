import torch

from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from display_utils import display_model
from pose_data import pose1,pose2,pose3,pose4


if __name__ == '__main__':
    cuda = False
    batch_size = 1

    # Create the SMPL layer
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender='female',
        model_root='smplpytorch/native/models')
    userInp = input("Please enter the desired pose (Hint: interger number from [1-4]  :  ")
    pose_params = None
    if userInp == '1':
    	pose_params = pose1
    elif userInp == '2':
    	pose_params = pose2
    elif userInp == '3':
    	pose_params = pose3
    elif userInp == '4':
    	pose_params = pose4
    else:
    	print("Please enter values between 1 to 4")
    	
   # Random preset shape
    shape_params =  torch.tensor([[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]) * 20

    # GPU mode
    if cuda:
        pose_params = pose_params.cuda()
        shape_params = shape_params.cuda()
        smpl_layer.cuda()

    # Forward from the SMPL layer
    if pose_params is not None:
        verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)
         # Draw output vertices and joints
        ax_r = display_model(
            {'verts': verts.cpu().detach(),
            'joints': Jtr.cpu().detach()},
            model_faces=smpl_layer.th_faces,
            with_joints=True,
            kintree_table=smpl_layer.kintree_table,
            savepath='image.png',
            show=True)
    else:
        print("exiting.....")
        exit

    



