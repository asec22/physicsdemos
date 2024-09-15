from flask import Flask, render_template, request, url_for, redirect
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from scipy.integrate import odeint
from matplotlib import rc

app = Flask(__name__)

form_dictionary = {
    "Linear Kinematics": ["Position (m)", "Velocity (m/s)", "Acceleration (m/s^2)", "Jerk (m/s^3)", "Snap (m/s^4)", "Crackle (m/s^5)", "Pop (m/s^6)"],
    "Projectile Demonstration": ["Velocity 1 (m/s)", "Velocity 2 (m/s)", "Angle 1 (degrees)", "Angle 2 (degrees)", 'X1 (m)', 'Y1 (m)', 'X2 (m)', 'Y2 (m)'],
    "Simple Pendulum": ['Length (m)','Initial Angle (degrees)', "Time Step (s))"],
    "Double Pendulum": ['Length 1 (m)', 'Length 2 (m)', 'Mass 1 (kg)','Mass 2 (kg)','Initial Angle 1 (degrees)', 'Initial Angle 2 (degrees)', 'Initial Angular Velocity 1 (m/s)', 'Angular Velocity 2 (m/s)'],
    'Inclined Plane':['Plane Angle (degrees, 0-80))','Initial Velocity (m/s)','Coefficient of Kinetic Friction']
}

description_dictionary = {"Linear Kinematics":"linear.txt", 
                          "Projectile Demonstration":"projectile.txt", 
                          "Simple Pendulum": "simplependulum.txt", 
                          "Double Pendulum":"doublependulum.txt", 
                          'Inclined Plane':'inclinedplane.txt'
}

demo_list = list(form_dictionary.keys())
g = 9.81

class Demonstration:
    def linearmotion(values):
        global lt,position,velocity,acceleration,jerk,snap,crackle,pop
        lt = np.linspace(0, 5, 40)
        framecount = len(lt)

        position = ((720 * values[6]) * lt ** 6) + ((120 * values[5]) * lt ** 5) + ((24 * values[4]) * lt ** 4) + ((6 * values[3]) * lt ** 3) + ((2 * values[2]) * lt ** 2) + ((values[1]) * lt) + values[0]
        velocity = ((120 * values[6]) * lt ** 5) + ((24 * values[5]) * lt ** 4) + ((6 * values[4]) * lt ** 3) + ((2 * values[3]) * lt ** 2) + ((values[2]) * lt) + values[1]
        acceleration = ((24 * values[6]) * lt ** 4) + ((6 * values[5]) * lt ** 3) + ((2 * values[4]) * lt ** 2) + (values[3] * lt) + values[2]
        jerk = ((6 * values[6]) * lt ** 3) + ((2 * values[5]) * lt ** 2) + ((values[4]) * lt) + values[3]
        snap = ((2 * values[6]) * lt ** 2) + (values[5] * lt) + values[4]
        crackle = (values[6] * lt) + values[5]
        pop = [values[6] for i in lt]

        max_list = [np.max(position), np.max(velocity), np.max(acceleration), np.max(jerk), np.max(snap), np.max(crackle), np.max(pop)]

        rc('animation', html='jshtml')
        fig, ax = plt.subplots()
        global posplot, velplot, acplot, jerkplot, snapplot, crackplot, popplot
        posplot = ax.plot(lt[0], position[0], label=f'Position ({values[0]} m)')[0]
        velplot = ax.plot(lt[0], velocity[0], label=f'Velocity ({values[1]} m/s)')[0]
        acplot = ax.plot(lt[0], acceleration[0], label=f'Acceleration ({values[2]} m/s^2)')[0]
        jerkplot = ax.plot(lt[0], jerk[0], label=f'Jerk ({values[3]} m/s^3)')[0]
        snapplot = ax.plot(lt[0], snap[0], label=f'Snap ({values[4]} m/s^4)')[0]
        crackplot = ax.plot(lt[0], crackle[0], label=f'Crackle ({values[5]} m/s^5)')[0]
        popplot = ax.plot(lt[0], pop[0], label=f'Pop ({values[6]} m/s^6)')[0]
        ax.set(xlim=[0, np.max(lt)], ylim=[0, (np.max(max_list) / 8)], xlabel='t [s]')
        ax.legend()
        return (fig,framecount)

    def projectiledemo(values):
        thetadeg1 = float(values[2])
        theta01 = np.radians(thetadeg1)
        vx01 = values[0] * np.cos(theta01)
        vy01 = values[0] * np.sin(theta01)
        yrange1 = ((vy01 ** 2) / (2 * g)) + values[6]
        trange1 = (vy01 / g) + (np.sqrt(((vy01 / g) ** 2) + ((2 * values[6]) / g)))
        xrange1 = (vx01 * trange1) + values[4]

        thetadeg2 = float(values[3])
        theta02 = np.radians(thetadeg2)
        vx02 = values[1] * np.cos(theta02)
        vy02 = values[1] * np.sin(theta02)
        yrange2 = ((vy02 ** 2) / (2 * g)) + values[7]
        trange2 = (vy02 / g) + (np.sqrt(((vy02 / g) ** 2) + ((2 * values[7]) / g)))
        xrange2 = (vx02 * trange2) + values[5]

        t = np.linspace(0, np.max([trange1, trange2]), 40)
        framecount = len(t)

        global x1, y1, x2, y2
        x1 = vx01 * t + values[4]
        y1 = -g * t ** 2 / 2 + vy01 * t + values[6]

        x2 = vx02 * t + values[5]
        y2 = -g * t ** 2 / 2 + vy02 * t + values[7]

        print(x1, y1, x2, y2)

        rc('animation', html='jshtml')
        fig, ax = plt.subplots()
        global projectile1, projectile2
        projectile1 = ax.scatter(x1[0], y1[0], c="b", s=5, label=f'v01 = {values[0]} m/s at {thetadeg1} degrees')
        projectile2 = ax.plot(x2[0], y2[0], label=f'v02 = {values[1]} m/s at {thetadeg2} degrees')[0]
        ax.set(xlim=[0, np.max([xrange1, xrange2])], ylim=[0, np.max([yrange1, yrange2])], xlabel='x [m]', ylabel='y [m]')
        ax.legend()
        return (fig,framecount)

    def pendplot(values):
        global pendx1, pendy1, pendx2, pendy2,framecount
        l = values[0]  # Length of the pendulum
        theta = values[1]  # Initial angle in degrees
        theta0 = np.radians(theta)  # Convert to radians

        w = np.sqrt(g / l)  # Angular frequency
        time_final = 4 * np.pi / w
        tp = np.arange(0, time_final, values[2])
        framecount = len(tp)  

        # Linear solution
        thetasol = theta0 * np.cos(w * tp)

        # Nonlinear solution using odeint
        x0 = 0.0  # Initial angular velocity
        initstate = [theta0, x0]
        theta1 = odeint(lambda state, t: [state[1], -(g/l)*np.sin(state[0])], initstate, tp)

        # Calculate x and y coordinates for both solutions
        pendx1 = l * np.sin(theta1[:, 0])
        pendy1 = -l * np.cos(theta1[:, 0])  # Negative to make the pendulum hang down

        pendx2 = l * np.sin(thetasol)
        pendy2 = -l * np.cos(thetasol)  # Negative to make the pendulum hang down


        # Set up the plot
        rc('animation', html='jshtml')
        fig, ax = plt.subplots()

        global pendline1, pendline2
        pendline1, = ax.plot([], [], 'o-', color="red", label="Nonlinear", lw=2)
        pendline2, = ax.plot([], [], 'o-', color="blue", label="Linear", lw=2)

        ax.set(xlim=[-l, l], ylim=[-l-1, 0.1*l], xlabel='x [m]', ylabel='y [m]')
        ax.legend()

        return (fig,framecount)
    
    def doublependulum(values):
        L1 = values[0]  # length of pendulum 1 in m
        L2 = values[1]  # length of pendulum 2 in m
        L = L1 + L2  # maximal length of the combined pendulum
        M1 = values[2]  # mass of pendulum 1 in kg
        M2 = values[3]  # mass of pendulum 2 in kg
        t_stop = 5.0  # how many seconds to simulate
        history_len = 500  # how many trajectory points to display


        def derivs(t, state):
            dydx = np.zeros_like(state)

            dydx[0] = state[1]

            delta = state[2] - state[0]
            den1 = (M1+M2) * L1 - M2 * L1 * np.cos(delta) * np.cos(delta)
            dydx[1] = ((M2 * L1 * state[1] * state[1] * np.sin(delta) * np.cos(delta)
                        + M2 * g * np.sin(state[2]) * np.cos(delta)
                        + M2 * L2 * state[3] * state[3] * np.sin(delta)
                        - (M1+M2) * g * np.sin(state[0]))
                    / den1)

            dydx[2] = state[3]

            den2 = (L2/L1) * den1
            dydx[3] = ((- M2 * L2 * state[3] * state[3] * np.sin(delta) * np.cos(delta)
                        + (M1+M2) * g * np.sin(state[0]) * np.cos(delta)
                        - (M1+M2) * L1 * state[1] * state[1] * np.sin(delta)
                        - (M1+M2) * g * np.sin(state[2]))
                    / den2)

            return dydx

        # create a time array from 0..t_stop sampled at 0.02 second steps
        global dt
        dt = 0.1
        tdp = np.arange(0, t_stop, dt)
        framecount = len(tdp)


        # th1 and th2 are the initial angles (degrees)
        # w10 and w20 are the initial angular velocities (degrees per second)
        th1 = values[4]
        w1 = values[6]
        th2 = values[5]
        w2 = values[7]

        # initial state
        state = np.radians([th1, w1, th2, w2])

        # integrate the ODE using Euler's method
        y = np.empty((len(tdp), 4))
        y[0] = state
        for i in range(1, len(tdp)):
            y[i] = y[i - 1] + derivs(tdp[i - 1], y[i - 1]) * dt

        # A more accurate estimate could be obtained e.g. using scipy:
        #
        #   y = scipy.integrate.solve_ivp(derivs, t[[0, -1]], state, t_eval=t).y.T

        global doublex1, doubley1, doublex2, doubley2
        doublex1 = L1*np.sin(y[:, 0])
        doubley1 = -L1*np.cos(y[:, 0])

        doublex2 = L2*np.sin(y[:, 2]) + doublex1
        doubley2 = -L2*np.cos(y[:, 2]) + doubley1

        fig = plt.figure(figsize=(5, 4))
        axes = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, L))
        axes.set_aspect('equal')
        axes.grid()

        global dpline, dptrace, time_template, time_text
        dpline, = axes.plot([], [], 'o-', lw=2)
        dptrace, = axes.plot([], [], '.-', lw=1, ms=2)
        time_template = 'time = %.1fs'
        time_text = axes.text(0.05, 0.9, '', transform=axes.transAxes)
        return (fig,framecount)
    
    def inclined_plane(values):
        global x_smooth, y_smooth, x_rough, y_rough, block_smooth, block_rough

        # Extract values from the input
        theta = np.radians(values[0])  # Angle of inclination in degrees
        v0 = values[1]  # Initial velocity in m/s
        mu_k = values[2]  # Coefficient of kinetic friction

        # Time array
        t_stop = 5  # Simulation time in seconds
        dt = 0.1  # Time step
        t = np.arange(0, t_stop, dt)
        framecount = len(t)

        # Smooth motion
        a_smooth = g * np.sin(theta)
        x_smooth = v0 * t + 0.5 * a_smooth * t**2
        x_smooth_max = v0 * t_stop + 0.5 * g*np.sin(np.radians(80)) *t_stop**2
        print(x_smooth_max,np.max(x_smooth))

        # Calculate y_smooth based on the incline
        y_smooth = - x_smooth * np.tan(theta)
        y_smooth_max = - x_smooth_max * np.tan(np.radians(80))
        print(y_smooth_max,np.min(y_smooth))
       

        # Rough motion
        a_rough = g * np.sin(theta) - mu_k * g * np.cos(theta)
        x_rough = v0 * t + 0.5 * a_rough * t**2
        

        # Calculate y_rough based on the incline
        y_rough = -x_rough * np.tan(theta)
    
         

        # Create the plot
        rc('animation', html='jshtml')
        fig, ax = plt.subplots(figsize=(5, 4))
        # Set the x and y limits based on the incline
        # Ensure limits include the entire motion
        ax.set_xlim([(np.min(x_smooth) - 0.1), (np.max(x_smooth) + 0.1)]) 
        ax.set_ylim([y_smooth_max, 0.0]) 
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Height (m)')
        ax.set_title('Block on Inclined Plane')

        # Plot the inclined plane correctly
        plane_x = [np.min(x_smooth), np.max(x_smooth)] 
        plane_y = [np.max(y_smooth), np.min(y_smooth)] 
        ax.plot(plane_x, plane_y, 'k-', linewidth=2)

        # Plot the blocks
        global block_smooth, block_rough
        block_smooth = ax.scatter(x_smooth[0], y_smooth[0], c="r", marker='s', s=50, label='Smooth')
        block_rough = ax.scatter(x_rough[0], y_rough[0], c="b", marker='s', s=50, label='Rough')

        # Add legend
        ax.legend()

        return (fig, framecount)


method_dictionary = {
    "Projectile Demonstration": Demonstration.projectiledemo,
    "Linear Kinematics": Demonstration.linearmotion,
    "Simple Pendulum": Demonstration.pendplot,
    "Double Pendulum": Demonstration.doublependulum,
    "Inclined Plane":Demonstration.inclined_plane
}

class Updates:
    def linearupdate(frame):
        posplot.set_data(lt[:frame], position[:frame])
        velplot.set_data(lt[:frame], velocity[:frame])
        acplot.set_data(lt[:frame], acceleration[:frame])
        jerkplot.set_data(lt[:frame], jerk[:frame])
        snapplot.set_data(lt[:frame], snap[:frame])
        crackplot.set_data(lt[:frame], crackle[:frame])
        popplot.set_data(lt[:frame], pop[:frame])
        return (posplot, velplot, acplot, snapplot, crackplot, popplot)

    def projectileupdate(frame):
        x = x1[:frame]
        y = y1[:frame]
        data = np.stack([x, y]).T
        projectile1.set_offsets(data)
        projectile2.set_xdata(x2[:frame])
        projectile2.set_ydata(y2[:frame])
        return (projectile1, projectile2)
    
    def simplepenupdate(frame ):  # Pass the instance as an argument
        thisx1 = [0, pendx1[frame]]
        thisy1 = [0, pendy1[frame]]
        thisx2 = [0, pendx2[frame]]
        thisy2 = [0, pendy2[frame]]
        pendline1.set_data(thisx1,thisy1)
        pendline2.set_data(thisx2,thisy2)

        return (pendline1, pendline2)
    
    def doublependulumupdate(frame):
        thisx = [0, doublex1[frame], doublex2[frame]]
        thisy = [0, doubley1[frame], doubley2[frame]]

        history_x = doublex2[:frame]
        history_y = doubley2[:frame]

        dpline.set_data(thisx, thisy)
        dptrace.set_data(history_x, history_y)
        time_text.set_text(time_template % (frame*dt))
        return dpline, dptrace, time_text
    
    def update_inclined_plane(frame):
        x1 = x_smooth[frame]
        y1 = y_smooth[frame]
        data1 = np.stack([x1, y1]).T
        x2 = x_rough[frame]
        y2 = y_rough[frame]
        data2 = np.stack([x2, y2]).T
        block_smooth.set_offsets(data1)
        block_rough.set_offsets(data2)
        return (block_smooth, block_rough)
    

update_dictionary = {
    "Projectile Demonstration": Updates.projectileupdate,
    "Linear Kinematics": Updates.linearupdate,
    "Simple Pendulum": Updates.simplepenupdate,
    "Double Pendulum": Updates.doublependulumupdate,
    'Inclined Plane': Updates.update_inclined_plane
}


def createani(dv, values):
    fig,framecount = method_dictionary[dv](values)
    ani = animation.FuncAnimation(fig, func=update_dictionary[dv], frames=framecount, interval=100)
    ani.save(filename=f'./static/{dv}plot.html', writer="html")
    plt.close()

@app.route("/")
def index():
    return render_template('index.html', dl=demo_list)

@app.route("/<demovalue>", methods=['POST', 'GET'])
def demo(demovalue):
    form_list = form_dictionary[demovalue]
    if request.method == "POST":
        values = [float(request.form[item]) for item in form_list]
        des=description_dictionary[demovalue]
        createani(demovalue, values)
        return redirect(url_for('plot', demovalue=demovalue))
    else:
        des=description_dictionary[demovalue]
        des_path=f'./static/descriptions/{des}'
        with open(des_path, 'r') as file:
            description = file.read()
        return render_template("form.html", dv=demovalue, fl=form_list,demo_description=description)

@app.route("/<demovalue>/plot")
def plot(demovalue):
    filename = f"{demovalue}plot.html"
    des=description_dictionary[demovalue]
    des_path=f'./static/descriptions/{des}'
    with open(des_path, 'r') as file:
        description = file.read()
    return render_template('plot.html', dv=demovalue, fn=filename,demo_description=description)

if __name__ == "__main__":
    app.run(debug=True)