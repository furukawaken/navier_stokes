#include "DataWriter.hpp"

const int NC_ERR = NC_EINVAL;

//------------NetCDFWriter------------//

NetCDFWriter::NetCDFWriter(
        const std::string& filename,
        const double dt, const double dy, const double dx,
        const int ny, const int nx):
    filename_(filename),
    ncid_(-1),
    time_dim_id_(-1), y_dim_id_(-1), x_dim_id_(-1),
    time_var_id_(-1), y_var_id_(-1), x_var_id_(-1),
    u_var_id_(-1),
    dt_(dt), dy_(dy), dx_(dx),
    ny_(ny),    nx_(nx),
    eta_(ny-1), xi_(nx-1) {}

NetCDFWriter::~NetCDFWriter() {
    //std::cout << "ncid_ = " << ncid_ << std::endl;
    //if (ncid_ != -1) {
    //    nc_sync(ncid_);
    //    nc_close(ncid_);
    //}
    std::cout << filename_ << " is closed." << std::endl;
    int retval = nc_sync(ncid_);
    if (retval != NC_NOERR) {
        std::cerr << "Error syncing file: " << nc_strerror(retval) << std::endl;
    }
    
    retval = nc_close(ncid_);
    if (retval != NC_NOERR) {
        std::cerr << "Error closing NetCDF file: " << nc_strerror(retval) << std::endl;
    }
}

void NetCDFWriter::reset() {
    if (ncid_ != -1) {
        ncid_ = -1;
    }
}

int NetCDFWriter::createFile() {
    reset();

    int retval;
    // Delete existing file if it exists
    std::ofstream ofile(filename_, std::ios::out | std::ios::trunc);
    if (ofile.is_open()) {
        ofile.close();
    }
    std::ifstream ifile(filename_);
    if (ifile) {
        ifile.close();
        if (std::remove(filename_.c_str()) != 0) {
            std::cerr << "Error deleting existing NetCDF file" << std::endl;
            return 1;
        } else {
            std::cout << "Existing NetCDF file deleted successfully." << std::endl;
        }
    }

    // Create a new NetCDF file
    retval = nc_create(filename_.c_str(), NC_NETCDF4, &ncid_);
    if (retval != NC_NOERR) return handleError(retval);
    
    // Define dimensions
    retval = nc_def_dim(ncid_, "x",    nx_,           &x_dim_id_);
    if (retval != NC_NOERR) return handleError(retval);
    retval = nc_def_dim(ncid_, "y",    ny_,           &y_dim_id_);
    if (retval != NC_NOERR) return handleError(retval);
    retval = nc_def_dim(ncid_, "time", NC_UNLIMITED, &time_dim_id_);
    if (retval != NC_NOERR) return handleError(retval);

    retval = nc_def_var(ncid_, "nt",      NC_DOUBLE, 1, &time_dim_id_, &time_var_id_); // time
    if (retval != NC_NOERR) return handleError(retval);
    retval = nc_def_var(ncid_, "x_coord", NC_DOUBLE, 1, &x_dim_id_,    &x_var_id_);
    if (retval != NC_NOERR) return handleError(retval);
    retval = nc_def_var(ncid_, "y_coord", NC_DOUBLE, 1, &y_dim_id_,    &y_var_id_);
    if (retval != NC_NOERR) return handleError(retval);
    // Define scalar field
    int dimids[3] = {time_dim_id_, y_dim_id_, x_dim_id_};
    retval = nc_def_var(ncid_, "u", NC_DOUBLE, 3, dimids, &u_var_id_);
    if (retval != NC_NOERR) return handleError(retval);

    retval = nc_enddef(ncid_);
    // End define mode

    return NC_NOERR;
}

int NetCDFWriter::writeTime(const double t, const int time_step) const {
    int retval;

    size_t time_var_start[1] = {static_cast<size_t>(time_step)}; // Start at current timestep
    size_t time_var_count[1] = {1};                              // Write 1 value (current time step)
    retval = nc_put_vara_double(ncid_, time_var_id_, time_var_start, time_var_count, &t);
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriter::writeData(const std::vector<double>& u, const int time_step, const std::string& variable_name) const {
    int retval;

    size_t u_var_start[3] = {static_cast<size_t>(time_step), 0, 0}; // Start at current timestep, entire x and y range
    size_t u_var_count[3] = {1, static_cast<size_t>(ny_), static_cast<size_t>(nx_)}; // Write 1 time step, all x, all y
    retval = nc_put_vara_double(ncid_, u_var_id_, u_var_start, u_var_count, u.data());
    if (retval != NC_NOERR) return handleError(retval);

    retval = nc_sync(ncid_);
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriter::setCoordinate() {
    int retval;
    // Create arrays for x and y coordinates
    std::vector<double> x_coords(nx_);
    std::vector<double> y_coords(ny_);

    // Fill x and y coordinates based on dx and dy
    for (int i = 0; i < nx_; ++i) {
        x_coords[i] = i * dx_;
    }
    for (int j = 0; j < ny_; ++j) {
        y_coords[j] = j * dy_;
    }

    // Write "x" coordinate data
    size_t x_start[1] = {0}; // Start from the beginning of the x dimension
    size_t x_count[1] = {static_cast<size_t>(nx_)}; // Number of x values
    retval = nc_put_vara_double(ncid_, x_var_id_, x_start, x_count, x_coords.data());
    if (retval != NC_NOERR) return handleError(retval);

    // Write "y" coordinate data
    size_t y_start[1] = {0}; // Start from the beginning of the y dimension
    size_t y_count[1] = {static_cast<size_t>(ny_)}; // Number of y values
    retval = nc_put_vara_double(ncid_, y_var_id_, y_start, y_count, y_coords.data());
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriter::handleError(int retval) const {
        std::cerr << "NetCDF error: " << nc_strerror(retval) << std::endl;
        return retval;
    }

//------------NetCDFWriterStokes------------//

NetCDFWriterStokes::NetCDFWriterStokes(
            const std::string& filename,
            const double dt, const double dy, const double dx,
                             const    int ny, const    int nx,
            const int nscalars
    ):
    filename_(filename),
    ncid_(-1),
    nscalars(nscalars),
    time_dim_id(-1), y_dim_id(-1),   x_dim_id(-1),
                     eta_dim_id(-1), xi_dim_id(-1),
    time_var_id(-1), y_var_id(-1),   x_var_id (-1),
    dt(dt),          dy(dy),         dx(dx),
                     ny(ny),         nx(nx),
                     neta(ny-1),     nxi(nx-1),
    u_var_id(-1), v_var_id(-1), p_var_id(-1),
    div_var_id(-1) {}

NetCDFWriterStokes::~NetCDFWriterStokes() {
    if (ncid_ != -1) {
        nc_close(ncid_);
        std::cout << filename_ << " is closed." << std::endl;
    }
}

void NetCDFWriterStokes::reset() {
    if (ncid_ != -1) {
        nc_close(ncid_);
        ncid_ = -1;
    }
}

int NetCDFWriterStokes::createFile() {
    reset();

    int retval;

    // Delete existing file if it exists
    std::ofstream ofile(filename_, std::ios::out | std::ios::trunc);
    if (ofile.is_open()) {
        ofile.close(); // 明示的に閉じる
    }
    std::ifstream ifile(filename_);
    if (ifile) {
        ifile.close();
        if (std::remove(filename_.c_str()) != 0) {
            std::cerr << "Error deleting existing NetCDF file" << std::endl;
            return 1;
        } else {
            std::cout << "Existing NetCDF file deleted successfully." << std::endl;
        }
    }

    // Create a new NetCDF file
    retval = nc_create(filename_.c_str(), NC_NETCDF4, &ncid_);
    if (retval != NC_NOERR) return handleError(retval);
    
    // Define dimensions
    retval = nc_def_dim(ncid_, "time", NC_UNLIMITED, &time_dim_id);
    if (retval != NC_NOERR) return handleError(retval);
    retval = nc_def_dim(ncid_, "y",    ny,           &y_dim_id);
    if (retval != NC_NOERR) return handleError(retval);
    retval = nc_def_dim(ncid_, "x",    nx,           &x_dim_id);
    if (retval != NC_NOERR) return handleError(retval);
    retval = nc_def_dim(ncid_, "eta",  neta,         &eta_dim_id);
    if (retval != NC_NOERR) return handleError(retval);
    retval = nc_def_dim(ncid_, "xi",   nxi,          &xi_dim_id);
    if (retval != NC_NOERR) return handleError(retval);

    // Define variable "time"
    retval = nc_def_var(ncid_, "t", NC_DOUBLE, 1, &time_dim_id, &time_var_id);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "y"
    retval = nc_def_var(ncid_, "y_coord",   NC_DOUBLE, 1, &y_dim_id,   &y_var_id);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "x"
    retval = nc_def_var(ncid_, "x_coord",   NC_DOUBLE, 1, &x_dim_id,   &x_var_id );
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "eta"
    retval = nc_def_var(ncid_, "eta_coord", NC_DOUBLE, 1, &eta_dim_id, &eta_var_id);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "xi"
    retval = nc_def_var(ncid_, "xi_coord",  NC_DOUBLE, 1, &xi_dim_id,  &xi_var_id);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "u"
    int dimids_u[3] = {time_dim_id, y_dim_id, xi_dim_id};
    retval = nc_def_var(ncid_, "u", NC_DOUBLE, 3, dimids_u, &u_var_id);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "v"
    int dimids_v[3] = {time_dim_id, eta_dim_id, x_dim_id};
    retval = nc_def_var(ncid_, "v", NC_DOUBLE, 3, dimids_v, &v_var_id);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "p"
    int dimids_p[3] = {time_dim_id, y_dim_id, x_dim_id};
    retval = nc_def_var(ncid_, "p", NC_DOUBLE, 3, dimids_p, &p_var_id);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "div"
    int dimids_div[3] = {time_dim_id, y_dim_id, x_dim_id};
    retval = nc_def_var(ncid_, "div", NC_DOUBLE, 3, dimids_div, &div_var_id);
    if (retval != NC_NOERR) return handleError(retval);

    // End define mode
    retval = nc_enddef(ncid_);
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriterStokes::writeTime(const double t, const int time_step) const {
    int retval;

    // Write "time" data
    size_t time_var_start[1] = {static_cast<size_t>(time_step)}; // Start at current timestep
    size_t time_var_count[1] = {1}; // Write 1 value (current time step)
    retval = nc_put_vara_double(ncid_, time_var_id, time_var_start, time_var_count, &t);
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriterStokes::writeVectorField(const std::vector<std::vector<double>>& uv, const int time_step) const {
    int retval;

    // Write "u" data
    retval = this->writeU(uv[0], time_step);
    retval = this->writeV(uv[1], time_step);
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriterStokes::writeU(const std::vector<double>& u, const int time_step) const {
    int retval;

    //std::cout << u[0] << " " << u[100] << " " << u[200] << " " << u[300] << std::endl; 

    // Write "u" data
    size_t u_var_start[3] = {static_cast<size_t>(time_step), 0, 0};
    size_t u_var_count[3] = {1, static_cast<size_t>(ny), static_cast<size_t>(nxi)};
    retval = nc_put_vara_double(ncid_, u_var_id, u_var_start, u_var_count, u.data());
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriterStokes::writeV(const std::vector<double>& v, const int time_step) const {
    int retval;

    //std::cout << v[0] << " " << v[100] << " " << v[200] << " " << v[300] << std::endl;

    // Write "v" data
    size_t v_var_start[3] = {static_cast<size_t>(time_step), 0, 0};
    size_t v_var_count[3] = {1, static_cast<size_t>(neta), static_cast<size_t>(nx)};
    retval = nc_put_vara_double(ncid_, v_var_id, v_var_start, v_var_count, v.data());
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriterStokes::writeP(const std::vector<double>& p, const int time_step) const {
    int retval;

    //std::cout << p[0] << " " << p[100] << " " << p[200] << " " << p[300] << std::endl;

    // Write "p" data
    size_t p_var_start[3] = {static_cast<size_t>(time_step), 0, 0};
    size_t p_var_count[3] = {1, static_cast<size_t>(ny), static_cast<size_t>(nx)};
    retval = nc_put_vara_double(ncid_, p_var_id, p_var_start, p_var_count, p.data());
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriterStokes::writeDiv(const std::vector<double>& div, const int time_step) const {
    int retval;

    //std::cout << div[0] << " " << div[100] << " " << div[200] << " " << div[300] << std::endl; 

    // Write "div" data
    size_t div_var_start[3] = {static_cast<size_t>(time_step), 0, 0};
    size_t div_var_count[3] = {1, static_cast<size_t>(ny), static_cast<size_t>(nx)};
    retval = nc_put_vara_double(ncid_, div_var_id, div_var_start, div_var_count, div.data());
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

// Save "an" scalar variable: (the variable would be "p", fixed later.)
int NetCDFWriterStokes::writeData(const std::vector<double>& data, const int time_step, const std::string& variable_name) const {
    int retval;

    // Function mapping:
    std::unordered_map<std::string, std::function<int(const std::vector<double>&, int)>> func_map = {
        {"u",   [this](const std::vector<double>& data, int time_step) { return this->writeU(  data, time_step); }},
        {"v",   [this](const std::vector<double>& data, int time_step) { return this->writeV(  data, time_step); }},
        {"p",   [this](const std::vector<double>& data, int time_step) { return this->writeP(  data, time_step); }},
        {"div", [this](const std::vector<double>& data, int time_step) { return this->writeDiv(data, time_step); }}
    };

    // Call function:
    auto it = func_map.find(variable_name);
    if (it != func_map.end()) {
        retval = it->second(data, time_step); // First: key, second: value
    } else {
        std::cerr << "Invalid variable name" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (retval != NC_NOERR) return handleError(retval);
    return NC_NOERR;
}

int NetCDFWriterStokes::setCoordinate() {
    int retval;
    // Create arrays for x and y coordinates
    std::vector<double> y_coords(ny);
    std::vector<double> x_coords(nx);
    std::vector<double> eta_coords(neta);
    std::vector<double> xi_coords(nxi);

    // Fill x and y coordinates based on dx and dy
    for (int jy = 0; jy < ny; ++jy) {
        y_coords[jy] = jy * dy;
    }
    for (int ix = 0; ix < nx; ++ix) {
        x_coords[ix] = ix * dx;
    }
    for (int jeta = 0; jeta < nxi; ++jeta) {
        eta_coords[jeta] = jeta * dy;
    }
    for (int ixi = 0; ixi < nx; ++ixi) {
        xi_coords[ixi] = ixi * dx;
    }

    // Write "y" coordinate data
    size_t y_start[1] = {0}; // Start from the beginning of the y dimension
    size_t y_count[1] = {static_cast<size_t>(ny)}; // Number of y values
    retval = nc_put_vara_double(ncid_, y_var_id, y_start, y_count, y_coords.data());
    if (retval != NC_NOERR) return handleError(retval);

    // Write "x" coordinate data
    size_t x_start[1] = {0}; // Start from the beginning of the x dimension
    size_t x_count[1] = {static_cast<size_t>(nx)}; // Number of x values
    retval = nc_put_vara_double(ncid_, x_var_id, x_start, x_count, x_coords.data());
    if (retval != NC_NOERR) return handleError(retval);

    // Write "eta" coordinate data
    size_t eta_start[1] = {0}; // Start from the beginning of the y dimension
    size_t eta_count[1] = {static_cast<size_t>(neta)}; // Number of y values
    retval = nc_put_vara_double(ncid_, eta_var_id, eta_start, eta_count, eta_coords.data());
    if (retval != NC_NOERR) return handleError(retval);

    // Write "xi" coordinate data
    size_t xi_start[1] = {0}; // Start from the beginning of the x dimension
    size_t xi_count[1] = {static_cast<size_t>(nxi)}; // Number of x values
    retval = nc_put_vara_double(ncid_, xi_var_id, xi_start, xi_count, xi_coords.data());
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

//------------NetCDFWriterTwoScalerFields------------//

NetCDFWriterTwoScalerFields::NetCDFWriterTwoScalerFields(
            const std::string& filename,
            const double dt, const double dy, const double dx,
                             const    int ny, const    int nx,
            const int nscalars
    ):
    filename_(filename),
    ncid_(-1),
    nscalars(nscalars),
    time_dim_id(-1), y_dim_id(-1),   x_dim_id(-1),
    time_var_id(-1), y_var_id(-1),   x_var_id (-1),
    dt(dt),          dy(dy),         dx(dx),
                     ny(ny),         nx(nx),
    u_var_id(-1), v_var_id(-1) {}

NetCDFWriterTwoScalerFields::~NetCDFWriterTwoScalerFields() {
    if (ncid_ != -1) {
        nc_close(ncid_);
        std::cout << filename_ << " is closed." << std::endl;
    }
}

void NetCDFWriterTwoScalerFields::reset() {
    if (ncid_ != -1) {
        nc_close(ncid_);
        ncid_ = -1;
    }
}

int NetCDFWriterTwoScalerFields::createFile() {
    reset();

    int retval;

    // Delete existing file if it exists
    std::ofstream ofile(filename_, std::ios::out | std::ios::trunc);
    if (ofile.is_open()) {
        ofile.close();
    }
    std::ifstream ifile(filename_);
    if (ifile) {
        ifile.close();
        if (std::remove(filename_.c_str()) != 0) {
            std::cerr << "Error deleting existing NetCDF file" << std::endl;
            return 1;
        } else {
            std::cout << "Existing NetCDF file deleted successfully." << std::endl;
        }
    }

    // Create a new NetCDF file
    retval = nc_create(filename_.c_str(), NC_NETCDF4, &ncid_);
    if (retval != NC_NOERR) return handleError(retval);
    
    // Define dimensions
    retval = nc_def_dim(ncid_, "time", NC_UNLIMITED, &time_dim_id);
    if (retval != NC_NOERR) return handleError(retval);
    retval = nc_def_dim(ncid_, "y",    ny,           &y_dim_id);
    if (retval != NC_NOERR) return handleError(retval);
    retval = nc_def_dim(ncid_, "x",    nx,           &x_dim_id);
    if (retval != NC_NOERR) return handleError(retval);

    // Define variable "time"
    retval = nc_def_var(ncid_, "t",       NC_DOUBLE, 1, &time_dim_id, &time_var_id);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "y"
    retval = nc_def_var(ncid_, "y_coord", NC_DOUBLE, 1, &y_dim_id,    &y_var_id);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "x"
    retval = nc_def_var(ncid_, "x_coord", NC_DOUBLE, 1, &x_dim_id,    &x_var_id );
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "u"
    int dimids_u[3] = {time_dim_id, y_dim_id, x_dim_id};
    retval = nc_def_var(ncid_, "u",       NC_DOUBLE, 3, dimids_u,     &u_var_id);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "v"
    int dimids_v[3] = {time_dim_id, y_dim_id, x_dim_id};
    retval = nc_def_var(ncid_, "v",       NC_DOUBLE, 3, dimids_v,     &v_var_id);
    if (retval != NC_NOERR) return handleError(retval);

    // End define mode
    retval = nc_enddef(ncid_);
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriterTwoScalerFields::setCoordinate() {
    int retval;
    // Create arrays for x and y coordinates
    std::vector<double> y_coords(ny);
    std::vector<double> x_coords(nx);

    // Fill x and y coordinates based on dx and dy
    for (int jy = 0; jy < ny; ++jy) {
        y_coords[jy] = jy * dy;
    }
    for (int ix = 0; ix < nx; ++ix) {
        x_coords[ix] = ix * dx;
    }

    // Write "y" coordinate data
    size_t y_start[1] = {0}; // Start from the beginning of the y dimension
    size_t y_count[1] = {static_cast<size_t>(ny)}; // Number of y values
    retval = nc_put_vara_double(ncid_, y_var_id, y_start, y_count, y_coords.data());
    if (retval != NC_NOERR) return handleError(retval);

    // Write "x" coordinate data
    size_t x_start[1] = {0}; // Start from the beginning of the x dimension
    size_t x_count[1] = {static_cast<size_t>(nx)}; // Number of x values
    retval = nc_put_vara_double(ncid_, x_var_id, x_start, x_count, x_coords.data());
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}


int NetCDFWriterTwoScalerFields::writeTime(const double t, const int time_step) const {
    int retval;

    // Write "time" data
    size_t time_var_start[1] = {static_cast<size_t>(time_step)}; // Start at current timestep
    size_t time_var_count[1] = {1}; // Write 1 value (current time step)
    retval = nc_put_vara_double(ncid_, time_var_id, time_var_start, time_var_count, &t);
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriterTwoScalerFields::writeVectorField(const std::vector<std::vector<double>>& uv, const int time_step) const {
    int retval;

    // Write "u" data
    retval = this->writeU(uv[0], time_step);
    retval = this->writeV(uv[1], time_step);
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriterTwoScalerFields::writeU(const std::vector<double>& u, const int time_step) const {
    int retval;

    //std::cout << u[0] << " " << u[100] << " " << u[200] << " " << u[300] << std::endl; 

    size_t u_var_start[3] = {static_cast<size_t>(time_step), 0, 0};
    size_t u_var_count[3] = {1, static_cast<size_t>(ny), static_cast<size_t>(nx)};
    retval = nc_put_vara_double(ncid_, u_var_id, u_var_start, u_var_count, u.data());
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriterTwoScalerFields::writeV(const std::vector<double>& v, const int time_step) const {
    int retval;

    //std::cout << v[0] << " " << v[100] << " " << v[200] << " " << v[300] << std::endl;

    // Write "v" data
    size_t v_var_start[3] = {static_cast<size_t>(time_step), 0, 0};
    size_t v_var_count[3] = {1, static_cast<size_t>(ny), static_cast<size_t>(nx)};
    retval = nc_put_vara_double(ncid_, v_var_id, v_var_start, v_var_count, v.data());
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

// Should be deleted later.
// Save "an" scalar variable: (the variable would be "p", fixed later.)
int NetCDFWriterTwoScalerFields::writeData(const std::vector<double>& data, const int time_step, const std::string& variable_name) const {
    int retval;

    // Function mapping:
    std::unordered_map<std::string, std::function<int(const std::vector<double>&, int)>> func_map = {
        {"u",   [this](const std::vector<double>& data, int time_step) { return this->writeU(  data, time_step); }},
        {"v",   [this](const std::vector<double>& data, int time_step) { return this->writeV(  data, time_step); }}
    };

    // Call function:
    auto it = func_map.find(variable_name);
    if (it != func_map.end()) {
        retval = it->second(data, time_step); // First: key, second: value
    } else {
        std::cerr << "Invalid variable name" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (retval != NC_NOERR) return handleError(retval);
    return NC_NOERR;
}

//------------NetCDFWriterNavierStokesPeriodic------------//

NetCDFWriterNavierStokesPeriodic::NetCDFWriterNavierStokesPeriodic(
            const std::string& filename,
            const double dt, const double dy, const double dx,
                             const    int ny, const    int nx,
            const double reynolds
    ):
    filename_(filename),
    ncid_(-1),
    time_dim_id(-1), y_dim_id_(-1),   x_dim_id_(-1),
                     eta_dim_id_(-1), xi_dim_id_(-1),
    time_var_id_(-1), y_var_id_(-1),   x_var_id_ (-1),
    dt_(dt),          dy_(dy),         dx_(dx),
                     ny_(ny),         nx_(nx),
    reynolds_(reynolds),
//                     neta(ny-1),     nxi(nx-1),
                     neta_(ny),     nxi_(nx),
    u_var_id_(-1), v_var_id_(-1), p_var_id_(-1)
    {}

NetCDFWriterNavierStokesPeriodic::~NetCDFWriterNavierStokesPeriodic() {
    if (ncid_ != -1) {
        nc_close(ncid_);
        std::cout << filename_ << " is closed." << std::endl;
    }
}

void NetCDFWriterNavierStokesPeriodic::reset() {
    if (ncid_ != -1) {
        nc_close(ncid_);
        ncid_ = -1;
    }
}

int NetCDFWriterNavierStokesPeriodic::createFile() {
    reset();

    int retval;

    // Delete existing file if it exists
    std::ofstream ofile(filename_, std::ios::out | std::ios::trunc);
    if (ofile.is_open()) {
        ofile.close();
    }
    std::ifstream ifile(filename_);
    if (ifile) {
        ifile.close();
        if (std::remove(filename_.c_str()) != 0) {
            std::cerr << "Error deleting existing NetCDF file" << std::endl;
            return 1;
        } else {
            std::cout << "Existing NetCDF file deleted successfully." << std::endl;
        }
    }

    // Create a new NetCDF file
    retval = nc_create(filename_.c_str(), NC_NETCDF4, &ncid_);
    if (retval != NC_NOERR) return handleError(retval);
    
    // Define dimensions
    retval = nc_def_dim(ncid_, "time", NC_UNLIMITED, &time_dim_id);
    if (retval != NC_NOERR) return handleError(retval);
    retval = nc_def_dim(ncid_, "y",    ny_,           &y_dim_id_);
    if (retval != NC_NOERR) return handleError(retval);
    retval = nc_def_dim(ncid_, "x",    nx_,           &x_dim_id_);
    if (retval != NC_NOERR) return handleError(retval);
    retval = nc_def_dim(ncid_, "eta",  neta_,         &eta_dim_id_);
    if (retval != NC_NOERR) return handleError(retval);
    retval = nc_def_dim(ncid_, "xi",   nxi_,          &xi_dim_id_);
    if (retval != NC_NOERR) return handleError(retval);

    // Define variable "time"
    retval = nc_def_var(ncid_, "time", NC_DOUBLE, 1, &time_dim_id, &time_var_id_);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "y"
    retval = nc_def_var(ncid_, "y",   NC_DOUBLE, 1, &y_dim_id_,   &y_var_id_);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "x"
    retval = nc_def_var(ncid_, "x",   NC_DOUBLE, 1, &x_dim_id_,   &x_var_id_ );
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "eta"
    retval = nc_def_var(ncid_, "eta_coord", NC_DOUBLE, 1, &eta_dim_id_, &eta_var_id_);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "xi"
    retval = nc_def_var(ncid_, "xi_coord",  NC_DOUBLE, 1, &xi_dim_id_,  &xi_var_id_);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "u"
    int dimids_u[3] = {time_dim_id, y_dim_id_, xi_dim_id_};
//    int dimids_u[3] = {time_dim_id, y_dim_id_, x_dim_id_};
    retval = nc_def_var(ncid_, "u", NC_DOUBLE, 3, dimids_u, &u_var_id_);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "v"
    int dimids_v[3] = {time_dim_id, eta_dim_id_, x_dim_id_};
//    int dimids_v[3] = {time_dim_id, y_dim_id_, x_dim_id_};
    retval = nc_def_var(ncid_, "v", NC_DOUBLE, 3, dimids_v, &v_var_id_);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "p"
    int dimids_p[3] = {time_dim_id, y_dim_id_, x_dim_id_};
    retval = nc_def_var(ncid_, "p", NC_DOUBLE, 3, dimids_p, &p_var_id_);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "optional"
    int dimids_opt[3] = {time_dim_id, y_dim_id_, x_dim_id_};
    retval = nc_def_var(ncid_, "optional", NC_DOUBLE, 3, dimids_opt, &opt_var_id_);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "optional2"
    int dimids_opt2[3] = {time_dim_id, y_dim_id_, x_dim_id_};
    retval = nc_def_var(ncid_, "optional2", NC_DOUBLE, 3, dimids_opt2, &opt_var_id2_);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "optional3"
    int dimids_opt3[3] = {time_dim_id, y_dim_id_, x_dim_id_};
    retval = nc_def_var(ncid_, "optional3", NC_DOUBLE, 3, dimids_opt3, &opt_var_id3_);
    if (retval != NC_NOERR) return handleError(retval);
    // Define variable "optional4"
    int dimids_opt4[3] = {time_dim_id, y_dim_id_, x_dim_id_};
    retval = nc_def_var(ncid_, "optional4", NC_DOUBLE, 3, dimids_opt4, &opt_var_id4_);
    if (retval != NC_NOERR) return handleError(retval);

    // Writing settings
    retval = nc_put_att_double(ncid_, NC_GLOBAL, "dt", NC_DOUBLE, 1, &dt_);
    if (retval != NC_NOERR) return handleError(retval);
    retval = nc_put_att_double(ncid_, NC_GLOBAL, "dx", NC_DOUBLE, 1, &dx_);
    if (retval != NC_NOERR) return handleError(retval);
    retval = nc_put_att_double(ncid_, NC_GLOBAL, "dy", NC_DOUBLE, 1, &dy_);
    if (retval != NC_NOERR) return handleError(retval);
    retval = nc_put_att_double(ncid_, NC_GLOBAL, "reynolds", NC_DOUBLE, 1, &reynolds_);
    if (retval != NC_NOERR) return handleError(retval);

    // End define mode
    retval = nc_enddef(ncid_);
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriterNavierStokesPeriodic::writeTime(const double t, const int time_step) const {
    int retval;

    size_t time_var_start[1] = {static_cast<size_t>(time_step)}; // Start at current timestep
    size_t time_var_count[1] = {1}; // Write 1 value (current time step)
    retval = nc_put_vara_double(ncid_, time_var_id_, time_var_start, time_var_count, &t);
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriterNavierStokesPeriodic::writeU(const std::vector<double>& u, const int time_step) const {
    int retval;
    size_t u_var_start[3] = {static_cast<size_t>(time_step), 0, 0};
    size_t u_var_count[3] = {1, static_cast<size_t>(ny_), static_cast<size_t>(nxi_)};
    //size_t u_var_count[3] = {1, static_cast<size_t>(ny_), static_cast<size_t>(nx)};
    retval = nc_put_vara_double(ncid_, u_var_id_, u_var_start, u_var_count, u.data());
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriterNavierStokesPeriodic::writeV(const std::vector<double>& v, const int time_step) const {
    int retval;
    size_t v_var_start[3] = {static_cast<size_t>(time_step), 0, 0};
    size_t v_var_count[3] = {1, static_cast<size_t>(neta_), static_cast<size_t>(nx_)};
    //size_t v_var_count[3] = {1, static_cast<size_t>(ny_), static_cast<size_t>(nx)};
    retval = nc_put_vara_double(ncid_, v_var_id_, v_var_start, v_var_count, v.data());
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriterNavierStokesPeriodic::writeP(const std::vector<double>& p, const int time_step) const {
    int retval;
    size_t p_var_start[3] = {static_cast<size_t>(time_step), 0, 0};
    size_t p_var_count[3] = {1, static_cast<size_t>(ny_), static_cast<size_t>(nx_)};
    retval = nc_put_vara_double(ncid_, p_var_id_, p_var_start, p_var_count, p.data());
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriterNavierStokesPeriodic::writeOptional(const std::vector<double>& opt, const int time_step) const {
    int retval;
    size_t opt_var_start[3] = {static_cast<size_t>(time_step), 0, 0};
    size_t opt_var_count[3] = {1, static_cast<size_t>(ny_), static_cast<size_t>(nx_)};
    retval = nc_put_vara_double(ncid_, opt_var_id_, opt_var_start, opt_var_count, opt.data());
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriterNavierStokesPeriodic::writeOptional2(const std::vector<double>& opt2, const int time_step) const {
    int retval;
    size_t opt_var2_start[3] = {static_cast<size_t>(time_step), 0, 0};
    size_t opt_var2_count[3] = {1, static_cast<size_t>(ny_), static_cast<size_t>(nx_)};
    retval = nc_put_vara_double(ncid_, opt_var_id2_, opt_var2_start, opt_var2_count, opt2.data());
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriterNavierStokesPeriodic::writeOptional3(const std::vector<double>& opt3, const int time_step) const {
    int retval;
    size_t opt_var3_start[3] = {static_cast<size_t>(time_step), 0, 0};
    size_t opt_var3_count[3] = {1, static_cast<size_t>(ny_), static_cast<size_t>(nx_)};
    retval = nc_put_vara_double(ncid_, opt_var_id3_, opt_var3_start, opt_var3_count, opt3.data());
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriterNavierStokesPeriodic::writeOptional4(const std::vector<double>& opt4, const int time_step) const {
    int retval;
    size_t opt_var4_start[3] = {static_cast<size_t>(time_step), 0, 0};
    size_t opt_var4_count[3] = {1, static_cast<size_t>(ny_), static_cast<size_t>(nx_)};
    retval = nc_put_vara_double(ncid_, opt_var_id4_, opt_var4_start, opt_var4_count, opt4.data());
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}

int NetCDFWriterNavierStokesPeriodic::setCoordinate() {
    int retval;
    // Create arrays for x and y coordinates
    std::vector<double> y_coords(ny_);
    std::vector<double> x_coords(nx_);
    std::vector<double> eta_coords(neta_);
    std::vector<double> xi_coords(nxi_);

    // Fill x and y coordinates based on dx and dy
    for (int jy = 0; jy < ny_; ++jy) {
        //y_coords[jy] = jy * dy;
        y_coords[jy] = (jy + 0.5) * dy_;
    }
    for (int ix = 0; ix < nx_; ++ix) {
        //x_coords[ix] = ix * dx;
        x_coords[ix] = (ix + 0.5) * dx_;
    }
    for (int jeta = 0; jeta < nxi_; ++jeta) {
        eta_coords[jeta] = jeta * dy_;
    }
    for (int ixi = 0; ixi < nx_; ++ixi) {
        xi_coords[ixi] = ixi * dx_;
    }

    // Write "y" coordinate data
    size_t y_start[1] = {0}; // Start from the beginning of the y dimension
    size_t y_count[1] = {static_cast<size_t>(ny_)}; // Number of y values
    retval = nc_put_vara_double(ncid_, y_var_id_, y_start, y_count, y_coords.data());
    if (retval != NC_NOERR) return handleError(retval);

    // Write "x" coordinate data
    size_t x_start[1] = {0}; // Start from the beginning of the x dimension
    size_t x_count[1] = {static_cast<size_t>(nx_)}; // Number of x values
    retval = nc_put_vara_double(ncid_, x_var_id_, x_start, x_count, x_coords.data());
    if (retval != NC_NOERR) return handleError(retval);

    // Write "eta" coordinate data
    size_t eta_start[1] = {0}; // Start from the beginning of the y dimension
    size_t eta_count[1] = {static_cast<size_t>(neta_)}; // Number of y values
    retval = nc_put_vara_double(ncid_, eta_var_id_, eta_start, eta_count, eta_coords.data());
    if (retval != NC_NOERR) return handleError(retval);

    // Write "xi" coordinate data
    size_t xi_start[1] = {0}; // Start from the beginning of the x dimension
    size_t xi_count[1] = {static_cast<size_t>(nxi_)}; // Number of x values
    retval = nc_put_vara_double(ncid_, xi_var_id_, xi_start, xi_count, xi_coords.data());
    if (retval != NC_NOERR) return handleError(retval);

    return NC_NOERR;
}
