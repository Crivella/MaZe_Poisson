import atexit
import os

num_threads = int(os.environ.get("OMP_NUM_THREADS", 1))


class OutputFiles:
    file_output_field = None
    file_output_performance = None
    file_output_iters = None
    file_output_energy = None
    file_output_temperature = None
    file_output_solute = None
    file_output_tot_force = None


open_files = []


def remove_existing_file(path: str):
    """If a file already exists at `path`, remove it safely."""
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"Removed existing file: {path}")
        except Exception as e:
            print(f"Failed to remove {path}: {e}")


def generate_output_file(out_path, overwrite=True):
    """Create (and register) an output file, overwriting if needed."""
    # Always remove the file if it already exists in the same directory
    if os.path.exists(out_path):
        if overwrite:
            remove_existing_file(out_path)
        else:
            raise ValueError(f"File {out_path} already exists")

    res = open(out_path, 'w')
    open_files.append(res)
    return res


def generate_output_files(grid, md_variables):
    N = grid.N
    N_p = grid.N_p
    output_settings = grid.output_settings
    md_variables = grid.md_variables
    path = output_settings.path

    output_files = OutputFiles()

    if md_variables.thermostat:
        thermostat_path = os.path.join(path, 'Thermostatted')
        os.makedirs(thermostat_path, exist_ok=True)
        path = thermostat_path

    # Helper function to create directories and clean files
    def prepare_output_file(filename):
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        remove_existing_file(full_path)
        return full_path

    if output_settings.print_field:
        output_field = prepare_output_file('field.csv')
        output_files.file_output_field = generate_output_file(output_field)
        output_files.file_output_field.write("iter,x,MaZe\n")

    if output_settings.print_performance:
        output_performance = prepare_output_file('performance.csv')
        output_files.file_output_performance = generate_output_file(output_performance)
        output_files.file_output_performance.write("iter,time,n_iters\n")

    if output_settings.print_iters:
        output_iters = prepare_output_file('iters.csv')
        output_files.file_output_iters = generate_output_file(output_iters)
        output_files.file_output_iters.write("iter,max_sigma,norm\n")

    if output_settings.print_energy:
        output_energy = prepare_output_file('energy.csv')
        output_files.file_output_energy = generate_output_file(output_energy)
        output_files.file_output_energy.write("iter,K,V_notelec\n")

    if output_settings.print_temperature:
        output_temperature = prepare_output_file('temperature.csv')
        output_files.file_output_temperature = generate_output_file(output_temperature)
        output_files.file_output_temperature.write("iter,T\n")

    if output_settings.print_solute:
        output_solute = prepare_output_file('solute.csv')
        output_files.file_output_solute = generate_output_file(output_solute)
        output_files.file_output_solute.write(
            "charge,iter,particle,x,y,z,vx,vy,vz,fx_elec,fy_elec,fz_elec\n"
        )

    if output_settings.print_tot_force:
        output_tot_force = prepare_output_file(f"tot_force_N{N}.csv")
        output_files.file_output_tot_force = generate_output_file(output_tot_force)
        output_files.file_output_tot_force.write("iter,Fx,Fy,Fz\n")

    return output_files


@atexit.register
def close_output_files():
    """Ensure all open output files are closed at program exit."""
    not_closed = []
    while open_files:
        app = open_files.pop()
        try:
            app.close()
        except Exception as e:
            print(f"Error closing file: {e}")
            not_closed.append(app)

    if not_closed:
        open_files.extend(not_closed)
        print("Some files could not be closed.")