#[derive(Debug)]
pub struct SiteProperties {
    pub vs30: f64,     // Site Vs30
    pub vs30_sim: f64, // Simulation Vs30
    pub pga: f64,      // Recorded PGA at site
}
