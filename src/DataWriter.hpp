#ifndef DATA_WRITER_HPP
#define DATA_WRITER_HPP

#include <fstream>
#include <functional>
#include <iostream>
#include <netcdf.h>
#include <stdexcept>
#include <utility>
#include <unordered_map>

class DataWriter {
    public:
        virtual ~DataWriter() = default;
        virtual int createFile() = 0;
        virtual int setCoordinate() = 0;
        virtual int writeTime(const double t, const int time_step) const = 0;
        virtual int writeData(
            const std::vector<double>& u,
            const int time_step=0,
            const std::string& variable_name="u"
        ) const = 0;
};

class NetCDFWriter : public DataWriter {
public:
    NetCDFWriter(const std::string& filename,
                   const double dt, const double dy, const double dx,
                   const int ny, const int nx
    );
    ~NetCDFWriter();
    void reset();
    int createFile() override;
    int setCoordinate() override;
    int writeTime(const double t, const int time_step) const override;
    int writeData(
        const std::vector<double>& u,
        const int time_step=0,
        const std::string& variable_name="u"
    ) const override;

private:
    std::string filename_;
    int ncid_;
    int x_dim_id_, y_dim_id_, time_dim_id_, u_var_id_, time_var_id_, x_var_id_, y_var_id_;
    double dt_, dy_, dx_;
    int nx_, ny_, eta_, xi_;

    int handleError(int retval) const ;
};

class NetCDFWriterStokes : public DataWriter {
public:
    NetCDFWriterStokes(
        const std::string& filename,
        const double dt, const double dy, const double dx,
                         const int    ny, const int    nx,
        const int nscalars=0
    );  
    ~NetCDFWriterStokes();
    void reset();
    int createFile() override;
    // Save "an" scalar variable: (the variable would be "p", fixed later.)
    int setCoordinate() override;
    int writeTime(const double t, const int time_step) const override;
    int writeVectorField(const std::vector<std::vector<double>>& uv, const int time_step) const;
    int writeU(  const std::vector<double>& u,   const int time_step) const;
    int writeV(  const std::vector<double>& v,   const int time_step) const;
    int writeP(  const std::vector<double>& p,   const int time_step) const;
    int writeDiv(const std::vector<double>& div, const int time_step) const;
    int writeData(
        const std::vector<double>& data,
        const int time_step=0,
        const std::string& variable_name="u"
    ) const override;

private:
    std::string filename_;
    int ncid_;
    int time_dim_id, y_dim_id,   x_dim_id;
    int xi_dim_id,   eta_dim_id;
    int time_var_id, y_var_id,   x_var_id;
    int              eta_var_id, xi_var_id;
    int u_var_id, v_var_id, p_var_id, div_var_id;
    double dt, dy, dx;
    int nx, ny, neta, nxi;
    int nscalars;

    int handleError(int retval) const {
        std::cerr << "NetCDF error: " << nc_strerror(retval) << std::endl;
        return retval;
    }
};

class NetCDFWriterTwoScalerFields : public DataWriter {
public:
    NetCDFWriterTwoScalerFields(
        const std::string& filename,
        const double dt, const double dy, const double dx,
                         const int    ny, const int    nx,
        const int nscalars=0
    );  
    ~NetCDFWriterTwoScalerFields();
    void reset();
    int createFile() override;
    // Save "an" scalar variable: (the variable would be "p", fixed later.)
    int setCoordinate() override;
    int writeTime(const double t, const int time_step) const override;
    int writeVectorField(const std::vector<std::vector<double>>& uv, const int time_step) const;
    int writeU(  const std::vector<double>& u,   const int time_step) const;
    int writeV(  const std::vector<double>& v,   const int time_step) const;
    int writeData(
        const std::vector<double>& data,
        const int time_step=0,
        const std::string& variable_name="u"
    ) const override;

private:
    std::string filename_;
    int ncid_;
    int time_dim_id, y_dim_id,   x_dim_id;
    int time_var_id, y_var_id,   x_var_id;
    int u_var_id, v_var_id;
    double dt, dy, dx;
    int nx, ny;
    int nscalars;

    int handleError(int retval) const {
        std::cerr << "NetCDF error: " << nc_strerror(retval) << std::endl;
        return retval;
    }
};

class NetCDFWriterNavierStokesPeriodic {
public:
    NetCDFWriterNavierStokesPeriodic(
        const std::string& filename,
        const double dt, const double dy, const double dx,
                         const int    ny, const int    nx,
        const double reynolds
    );
    ~NetCDFWriterNavierStokesPeriodic();
    void reset();
    int createFile();
    int setCoordinate();
    int writeTime(const double t, const int time_step) const;
    int writeU(   const std::vector<double>& u,   const int time_step) const;
    int writeV(   const std::vector<double>& v,   const int time_step) const;
    int writeP(   const std::vector<double>& p,   const int time_step) const;
    int writeOptional(const std::vector<double>& opt, const int time_step) const;
    int writeOptional2(const std::vector<double>& opt2, const int time_step) const;
    int writeOptional3(const std::vector<double>& opt3, const int time_step) const;
    int writeOptional4(const std::vector<double>& opt4, const int time_step) const;

private:
    std::string filename_;
    int ncid_;
    int time_dim_id, y_dim_id_,   x_dim_id_;
    int xi_dim_id_,   eta_dim_id_;
    int time_var_id_, y_var_id_,   x_var_id_;
    int              eta_var_id_, xi_var_id_;
    int u_var_id_, v_var_id_, p_var_id_, opt_var_id_, opt_var_id2_, opt_var_id3_, opt_var_id4_;
    double dt_, dy_, dx_;
    double reynolds_;
    int nx_, ny_;
    int neta_, nxi_;

    int handleError(int retval) const {
        std::cerr << "NetCDF error: " << nc_strerror(retval) << std::endl;
        return retval;
    }
};

#endif // DATA_WRITER_HPP