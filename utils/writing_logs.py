def writing_logs(writer, train_metrics, val_metrics, epoch):
    """-------------------------Loss--------------------------------------------------------"""               
    writer.add_scalars('Loss/mean',
                        {'train':train_metrics['Loss'][3],
                        'Val':val_metrics['Loss'][3]},
                        epoch)
                        
    writer.add_scalars('Loss/Aorta',
                        {'train':train_metrics['Loss'][0],
                            'Val':val_metrics['Loss'][0]},
                            epoch)
    
    writer.add_scalars('Loss/Gallbladder',
                        {'train':train_metrics['Loss'][1],
                            'Val':val_metrics['Loss'][1]},
                            epoch)
    
    writer.add_scalars('Loss/Spleen',
                        {'train':train_metrics['Loss'][2],
                            'Val':val_metrics['Loss'][2]},
                            epoch)
    writer.add_scalars('Loss/Left_Kidney',
                        {'train':train_metrics['Loss'][3],
                            'Val':val_metrics['Loss'][3]},
                            epoch)
    writer.add_scalars('Loss/Right_Kidney',
                        {'train':train_metrics['Loss'][4],
                            'Val':val_metrics['Loss'][4]},
                            epoch)
    writer.add_scalars('Loss/Liver',
                        {'train':train_metrics['Loss'][5],
                            'Val':val_metrics['Loss'][5]},
                            epoch)
    writer.add_scalars('Loss/Pancreas',
                        {'train':train_metrics['Loss'][6],
                            'Val':val_metrics['Loss'][6]},
                            epoch)
    writer.add_scalars('Loss/Stomach',
                        {'train':train_metrics['Loss'][7],
                            'Val':val_metrics['Loss'][7]},
                            epoch)
                    
    """-------------------------Dice--------------------------------------------------------"""
    writer.add_scalars('Dice/mean',
                        {'train':train_metrics['Dice'][3],
                            'Val':val_metrics['Dice'][3]},
                            epoch)

    writer.add_scalars('Dice/Aorta',
                        {'train':train_metrics['Dice'][0],
                            'Val':val_metrics['Dice'][0]},
                            epoch)
    
    writer.add_scalars('Dice/Gallbladder',
                        {'train':train_metrics['Dice'][1],
                            'Val':val_metrics['Dice'][1]},
                            epoch)
    
    writer.add_scalars('Dice/Spleen',
                        {'train':train_metrics['Dice'][2],
                            'Val':val_metrics['Dice'][2]},
                            epoch)
    writer.add_scalars('Dice/Left_Kidney',
                        {'train':train_metrics['Dice'][3],
                            'Val':val_metrics['Dice'][3]},
                            epoch)
    writer.add_scalars('Dice/Right_Kidney',
                        {'train':train_metrics['Dice'][4],
                            'Val':val_metrics['Dice'][4]},
                            epoch)
    writer.add_scalars('Dice/Liver',
                        {'train':train_metrics['Dice'][5],
                            'Val':val_metrics['Dice'][5]},
                            epoch)
    writer.add_scalars('Dice/Pancreas',
                        {'train':train_metrics['Dice'][6],
                            'Val':val_metrics['Dice'][6]},
                            epoch)
    writer.add_scalars('Dice/Stomach',
                        {'train':train_metrics['Dice'][7],
                            'Val':val_metrics['Dice'][7]},
                            epoch)
    
    """-------------------------Precision--------------------------------------------------------"""
    writer.add_scalars('Precision/mean',
                        {'train':train_metrics['Precision'][3],
                            'Val':val_metrics['Precision'][3]},
                            epoch)

    writer.add_scalars('Precision/Aorta',
                        {'train':train_metrics['Precision'][0],
                            'Val':val_metrics['Precision'][0]},
                            epoch)
    
    writer.add_scalars('Precision/Gallbladder',
                        {'train':train_metrics['Precision'][1],
                            'Val':val_metrics['Precision'][1]},
                            epoch)
    
    writer.add_scalars('Precision/Spleen',
                        {'train':train_metrics['Precision'][2],
                            'Val':val_metrics['Precision'][2]},
                            epoch)
    writer.add_scalars('Precision/Left_Kidney',
                        {'train':train_metrics['Precision'][3],
                            'Val':val_metrics['Precision'][3]},
                            epoch)
    writer.add_scalars('Precision/Right_Kidney',
                        {'train':train_metrics['Precision'][4],
                            'Val':val_metrics['Precision'][4]},
                            epoch)
    writer.add_scalars('Precision/Liver',
                        {'train':train_metrics['Precision'][5],
                            'Val':val_metrics['Precision'][5]},
                            epoch)
    writer.add_scalars('Precision/Pancreas',
                        {'train':train_metrics['Precision'][6],
                            'Val':val_metrics['Precision'][6]},
                            epoch)
    writer.add_scalars('Precision/Stomach',
                        {'train':train_metrics['Precision'][7],
                            'Val':val_metrics['Precision'][7]},
                            epoch)
    
    """-------------------------Recall--------------------------------------------------------"""
    writer.add_scalars('Recall/mean',
                        {'train':train_metrics['Recall'][3],
                            'Val':val_metrics['Recall'][3]},
                            epoch)
    
    writer.add_scalars('Recall/Aorta',
                        {'train':train_metrics['Recall'][0],
                            'Val':val_metrics['Recall'][0]},
                            epoch)
    
    writer.add_scalars('Recall/Gallbladder',
                        {'train':train_metrics['Recall'][1],
                            'Val':val_metrics['Recall'][1]},
                            epoch)
    
    writer.add_scalars('Recall/Spleen',
                        {'train':train_metrics['Recall'][2],
                            'Val':val_metrics['Recall'][2]},
                            epoch)
    writer.add_scalars('Recall/Left_Kidney',
                        {'train':train_metrics['Recall'][3],
                            'Val':val_metrics['Recall'][3]},
                            epoch)
    writer.add_scalars('Recall/Right_Kidney',
                        {'train':train_metrics['Recall'][4],
                            'Val':val_metrics['Recall'][4]},
                            epoch)
    writer.add_scalars('Recall/Liver',
                        {'train':train_metrics['Recall'][5],
                            'Val':val_metrics['Recall'][5]},
                            epoch)
    writer.add_scalars('Recall/Pancreas',
                        {'train':train_metrics['Recall'][6],
                            'Val':val_metrics['Recall'][6]},
                            epoch)
    writer.add_scalars('Recall/Stomach',
                        {'train':train_metrics['Recall'][7],
                            'Val':val_metrics['Recall'][7]},
                            epoch)
    
    """-------------------------F1_scores--------------------------------------------------------"""
    writer.add_scalars('F1_scores/mean',
                        {'train':train_metrics['F1_scores'][3],
                            'Val':val_metrics['F1_scores'][3]},
                            epoch)
    
    writer.add_scalars('F1_scores/Aorta',
                        {'train':train_metrics['F1_scores'][0],
                            'Val':val_metrics['F1_scores'][0]},
                            epoch)
    
    writer.add_scalars('F1_scores/Gallbladder',
                        {'train':train_metrics['F1_scores'][1],
                            'Val':val_metrics['F1_scores'][1]},
                            epoch)
    
    writer.add_scalars('F1_scores/Spleen',
                        {'train':train_metrics['F1_scores'][2],
                            'Val':val_metrics['F1_scores'][2]},
                            epoch)
    writer.add_scalars('F1_scores/Left_Kidney',
                        {'train':train_metrics['F1_scores'][3],
                            'Val':val_metrics['F1_scores'][3]},
                            epoch)
    writer.add_scalars('F1_scores/Right_Kidney',
                        {'train':train_metrics['F1_scores'][4],
                            'Val':val_metrics['F1_scores'][4]},
                            epoch)
    writer.add_scalars('F1_scores/Liver',
                        {'train':train_metrics['F1_scores'][5],
                            'Val':val_metrics['F1_scores'][5]},
                            epoch)
    writer.add_scalars('F1_scores/Pancreas',
                        {'train':train_metrics['F1_scores'][6],
                            'Val':val_metrics['F1_scores'][6]},
                            epoch)
    writer.add_scalars('F1_scores/Stomach',
                        {'train':train_metrics['F1_scores'][7],
                            'Val':val_metrics['F1_scores'][7]},
                            epoch)
    
    """-------------------------mIoU--------------------------------------------------------"""
    writer.add_scalars('mIoU/mean',
                        {'train':train_metrics['mIoU'][3],
                            'Val':val_metrics['mIoU'][3]},
                            epoch)
    
    writer.add_scalars('mIoU/Aorta',
                        {'train':train_metrics['mIoU'][0],
                            'Val':val_metrics['mIoU'][0]},
                            epoch)
    
    writer.add_scalars('mIoU/Gallbladder',
                        {'train':train_metrics['mIoU'][1],
                            'Val':val_metrics['mIoU'][1]},
                            epoch)
    
    writer.add_scalars('mIoU/Spleen',
                        {'train':train_metrics['mIoU'][2],
                            'Val':val_metrics['mIoU'][2]},
                            epoch)
    
    writer.add_scalars('mIoU/Left_Kidney',
                        {'train':train_metrics['mIoU'][3],
                            'Val':val_metrics['mIoU'][3]},
                            epoch)
    writer.add_scalars('mIoU/Right_Kidney',
                        {'train':train_metrics['mIoU'][4],
                            'Val':val_metrics['mIoU'][4]},
                            epoch)
    writer.add_scalars('mIoU/Liver',
                        {'train':train_metrics['mIoU'][5],
                            'Val':val_metrics['mIoU'][5]},
                            epoch)
    writer.add_scalars('mIoU/Pancreas',
                        {'train':train_metrics['mIoU'][6],
                            'Val':val_metrics['mIoU'][6]},
                            epoch)
    writer.add_scalars('mIoU/Stomach',
                        {'train':train_metrics['mIoU'][7],
                            'Val':val_metrics['mIoU'][7]},
                            epoch)
    
    """-------------------------Accuracy--------------------------------------------------------"""
    writer.add_scalars('Accuracy/mean',
                        {'train':train_metrics['Accuracy'][3],
                            'Val':val_metrics['Accuracy'][3]},
                            epoch)
    
    writer.add_scalars('Accuracy/Aorta',
                        {'train':train_metrics['Accuracy'][0],
                            'Val':val_metrics['Accuracy'][0]},
                            epoch)
    
    writer.add_scalars('Accuracy/Gallbladder',
                        {'train':train_metrics['Accuracy'][1],
                            'Val':val_metrics['Accuracy'][1]},
                            epoch)
    
    writer.add_scalars('Accuracy/Spleen',
                        {'train':train_metrics['Accuracy'][2],
                            'Val':val_metrics['Accuracy'][2]},
                            epoch)
    
    writer.add_scalars('Accuracy/Left_Kidney',
                        {'train':train_metrics['Accuracy'][3],
                            'Val':val_metrics['Accuracy'][3]},
                            epoch)
    writer.add_scalars('Accuracy/Right_Kidney',
                        {'train':train_metrics['Accuracy'][4],
                            'Val':val_metrics['Accuracy'][4]},
                            epoch)
    writer.add_scalars('Accuracy/Liver',
                        {'train':train_metrics['Accuracy'][5],
                            'Val':val_metrics['Accuracy'][5]},
                            epoch)
    writer.add_scalars('Accuracy/Pancreas',
                        {'train':train_metrics['Accuracy'][6],
                            'Val':val_metrics['Accuracy'][6]},
                            epoch)
    writer.add_scalars('Accuracy/Stomach',
                        {'train':train_metrics['Accuracy'][7],
                            'Val':val_metrics['Accuracy'][7]},
                            epoch)

            