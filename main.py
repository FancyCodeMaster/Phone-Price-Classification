from flask import Flask , render_template , request
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route("/" , methods=["GET" , "POST"])
def main():
    if request.method == "POST":
        battery_power = request.form.get("battery_power")
        bluetooth = request.form.get("blue")
        clock_speed = request.form.get("clock_speed")
        dual_sim = request.form.get("dual_sim")
        front_camera = request.form.get("fc")
        four_g = request.form.get("four_g")
        internal_memory = request.form.get("int_memory")
        mobile_depth = request.form.get("m_dep")
        mobile_weight = request.form.get("mobile_wt")
        number_of_cores = request.form.get("n_cores")
        primary_camera = request.form.get("pc")
        pixel_height = request.form.get("px_height")
        pixel_width = request.form.get("px_width")
        ram = request.form.get("ram")
        screen_height = request.form.get("sc_h")
        screen_width = request.form.get("sc_w")
        talk_time = request.form.get("talk_time")
        three_g = request.form.get("three_g")
        touch_screen = request.form.get("touch_screen")
        wifi = request.form.get("wifi")

        # return str(battery_power) + str(bluetooth) + str(clock_speed)+ str(dual_sim) +  str(front_camera) + str(four_g) + str(internal_memory) + str(mobile_depth) + str(mobile_weight) + str(number_of_cores) + str(primary_camera) + str(pixel_height) + str(pixel_width) +  str(ram) + str(screen_height) + str(screen_width) + str(talk_time) + str(three_g) + str(touch_screen) + str(wifi)

        arr = np.array([battery_power , bluetooth,clock_speed,dual_sim,front_camera,four_g,internal_memory,mobile_depth,mobile_weight,number_of_cores,primary_camera,pixel_height,pixel_width,ram,screen_height,screen_width,talk_time,three_g,touch_screen,wifi]).reshape((1,20))
        arr_scaler = StandardScaler()
        arr = arr_scaler.fit_transform(arr)
        log_reg_model = joblib.load("logisticRegression.pkl")

        try:
            return render_template('result.html' , data=str(log_reg_model.predict(arr)))
        except:
            return 'just wait'
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug = True)