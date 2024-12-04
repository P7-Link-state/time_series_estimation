from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def produce_dataset(obj_act, validation_split=0.1, pred_range=10):
    
    pred_range = int(pred_range)  # Prediction range in seconds must be an integer.
    # Initialize lists for features (X) and targets (y)
    X = []
    y = []

    X_val=[]
    y_val=[]

    importance = []
    importance_val = []
    # validation_split=0.1 # 20 percent for validation
    n_train_test_sets=int(len(obj_act)*(1-validation_split))+1
    n_validation_sets=len(obj_act)-int(len(obj_act)*(1-validation_split))-1
    # print(f"train/test sets = {int(len(obj_act)*(1-validation_split))+1}"+f", validation sets = {len(obj_act)-int(len(obj_act)*(1-validation_split))-1}")

    for i in range(len(obj_act)):
        prediction_time = np.arange(int(obj_act[i].time_ax[0+20])+10, int(obj_act[i].time_ax[-1]) + 1, 1)  # Prediction time points from just after the 20 first point to the last point

        # Initialize arrays for linear regression
        n_points_fit = np.zeros((len(prediction_time)))
        n_points = np.zeros((len(prediction_time)-pred_range))
        # coef = np.zeros((len(prediction_time)))
        # intercept = np.zeros((len(prediction_time)))
        slope_val = np.zeros((len(prediction_time)))
        mean = np.zeros((len(prediction_time)))
        mse_slope = np.zeros((len(prediction_time)))

        # Initialize arrays for polynomial regression
        # poly_coef = np.zeros((len(prediction_time), 3))  # Storing 3 coefficients for the 2nd-degree polynomial
        poly_val = np.zeros((len(prediction_time)))      # Predicted values for the polynomial
        mse_poly = np.zeros((len(prediction_time)))

        #all the classic sheit
        azi = np.zeros((len(prediction_time)))
        ele = np.zeros((len(prediction_time)))
        pe = np.zeros((len(prediction_time)))
        fspl = np.zeros((len(prediction_time)))
        noise = np.zeros((len(prediction_time)))
        x_time = np.zeros((len(prediction_time)))
        target = np.zeros((len(prediction_time)-pred_range))


        # print(f"Prediction time: {prediction_time}")

        for q, time in enumerate(prediction_time):
            # Ensure time_ax is a numpy array
            time_ax = np.asarray(obj_act[i].time_ax)
            
            # Get indices for the specified range
            indices = np.where((time_ax > time - (20+pred_range)) & (time_ax < time - pred_range)) #20 seconds before the prediction time
            n_points_fit[q] = indices[0].size
            # print(f"Indices len: {n_points_fit[q]}")
            if indices[0].size == 0:
                pass
            else:
                x_lin = np.array(obj_act[i].time_ax[indices[0]]).reshape(-1, 1)  # Reshape for sklearn
                y_lin = np.array(obj_act[i].clean_sig_abs[indices[0]])  # Dependent variable (target)
                
                # Compute weights: closer to `time` has higher weight
                distances = np.abs(x_lin.flatten() - time)  # Distance from current `time`
                weights = np.exp(-distances / 10)  # Exponential decay (adjust scale as needed)

                mean[q] = np.mean(y_lin)

                #All the classic sheit
                azi[q] =   obj_act[i].station_obj.azimuth[indices[0][-1]]
                ele[q] =   obj_act[i].station_obj.elevation[indices[0][-1]]
                pe[q] =    np.mean(10*np.log10(obj_act[i].pointing_error[indices[0]])*weights)
                fspl[q] =  obj_act[i].station_obj.fspl[indices[0][-1]]
                noise[q] = obj_act[i].noise_obj.noise[indices[0][-1]]
                x_time[q]= obj_act[i].time_ax[indices[0][-1]]
                
                if q>=pred_range:
                    target[q-pred_range] = obj_act[i].target[indices[0][-1]] #saving the target of the prediction
                    n_points[q-pred_range] = np.where((time_ax > time - (1+pred_range)) & (time_ax < time - pred_range))[0].size #How many points are the one second of the prediction time means how important each point is

                if indices[0].size > 10: #choose how many samples are needed for the regression to be valid
                    # --- Linear Regression ---
                    model = LinearRegression()
                    model.fit(x_lin, y_lin, sample_weight=weights)
                    
                    # Save model outputs
                    slope_val[q] = model.coef_[0] * (time + pred_range) + model.intercept_
                    y_pred = model.predict(x_lin)
                    mse_slope[q] = mean_squared_error(y_lin, y_pred)

                    # Polynomial Regression
                    x_lin_flat = x_lin.flatten()  # Flatten for numpy.polyfit
                    poly = np.polyfit(x_lin_flat, y_lin, 2, w=weights)  # 2nd-degree polynomial with weights

                    # Polynomial prediction for the same points
                    poly_pred = np.polyval(poly, x_lin_flat)
                    poly_val[q] = np.polyval(poly, time + pred_range)  # Prediction at (time + pred_range)
                    mse_poly[q] = mean_squared_error(y_lin, poly_pred)
                else:
                    slope_val[q] = np.mean(y_lin) #If there are not enough samples, just take the mean
                    poly_val[q] = np.mean(y_lin) #If there are not enough samples, just take the mean
                    


                

        features = np.stack([
            azi   ,
            ele   ,
            pe    ,
            fspl  ,
            noise ,
            x_time,
            mean,
            slope_val,
            mse_slope,
            poly_val,
            mse_poly,
            n_points_fit
        ], axis=1)

        
        # Slice the array, ensuring that it remains 2D
        valid_indices = np.where((n_points > 0) & (target > 0))[0] # delete all the targets that have less than 1 points and if target is 0, as they are NOT important
        # print(f"Valid indices: {valid_indices}")
        features = features[valid_indices,:] 
        target = target[valid_indices] 
        n_points = n_points[valid_indices]


        if i < len(obj_act)*(1-validation_split):
            X.append(features)
            y.append(target)
            importance.append(n_points)
        else:
            X_val.append(features)
            y_val.append(target)
            importance_val.append(n_points)    

    X = np.vstack(X)
    y = np.hstack(y)
    X = np.squeeze(X)
    # print(f"X shape: {X.shape}")

    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    X_val_normalized = []
    for i in range(len(X_val)):
        X_val_normalized.append(scaler.transform(X_val[i]))
        # print(f"X_val shape: {len(X_val[i])}")
    
    scaler = MinMaxScaler()
    importance=np.hstack(importance)
    importance = np.array(scaler.fit_transform(importance.reshape(-1, 1))).flatten()
    importance_val_normalized = []
    # print(f"importance_val values: {importance}")
    
    for i in range(len(importance_val)):
        # print(f"importance_val values: {importance_val[i].reshape(-1, 1)}")
        importance_val_normalized.append(scaler.transform(importance_val[i].reshape(-1, 1)))
    # Convert to NumPy arrays for input to the model
    y = np.array(y)
    # print(f"importance_val_normalized shape: {len(importance_val_normalized)}")

    return X_normalized, y, X_val, X_val_normalized, y_val, n_train_test_sets, n_validation_sets, importance, importance_val_normalized