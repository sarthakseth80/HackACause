"import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Map, 
  Truck, 
  Droplets,
  AlertTriangle,
  CloudRain,
  Gauge,
  TrendingUp,
  ListOrdered,
  Route,
  Activity,
  Menu,
  X
} from 'lucide-react';
import { useState } from 'react';

const navItems = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard, section: 'main' },
  { path: '/map', label: 'Map View', icon: Map, section: 'main' },
  { path: '/villages', label: 'Villages', icon: Droplets, section: 'main' },
  { path: '/tankers', label: 'Tanker Management', icon: Truck, section: 'main' },
  // Analysis Modules
  { path: '/rainfall-analysis', label: 'Rainfall Analyzer', icon: CloudRain, section: 'analysis' },
  { path: '/groundwater-analysis', label: 'Groundwater Analyzer', icon: Droplets, section: 'analysis' },
  { path: '/stress-index', label: 'Stress Index', icon: Gauge, section: 'analysis' },
  { path: '/demand-predictor', label: 'Demand Predictor', icon: TrendingUp, section: 'analysis' },
  { path: '/priority-allocation', label: 'Priority Allocation', icon: ListOrdered, section: 'analysis' },
  { path: '/route-optimization', label: 'Route Optimization', icon: Route, section: 'analysis' },
  { path: '/realtime-monitoring', label: 'Real-Time Monitor', icon: Activity, section: 'analysis' },
];

export const Sidebar = () => {
  const [isOpen, setIsOpen] = useState(false);

  const mainItems = navItems.filter(i => i.section === 'main');
  const analysisItems = navItems.filter(i => i.section === 'analysis');

  return (
    <>
      {/* Mobile menu button */}
      <button
        data-testid=\"mobile-menu-btn\"
        onClick={() => setIsOpen(!isOpen)}
        className=\"lg:hidden fixed top-4 left-4 z-50 p-2 bg-slate-900 text-white rounded-lg shadow-lg\"
      >
        {isOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      {/* Overlay for mobile */}
      {isOpen && (
        <div 
          className=\"lg:hidden fixed inset-0 bg-black/50 z-40\"
          onClick={() => setIsOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside 
        data-testid=\"sidebar\"
        className={`sidebar transform transition-transform duration-300 overflow-y-auto ${
          isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        {/* Logo */}
        <div className=\"p-6 border-b border-slate-800\">
          <div className=\"flex items-center gap-3\">
            <div className=\"w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center\">
              <Droplets className=\"w-6 h-6 text-white\" />
            </div>
            <div>
              <h1 className=\"text-xl font-bold tracking-tight\" style={{ fontFamily: 'Manrope' }}>
                DroughtGuard
              </h1>
              <p className=\"text-xs text-slate-400\">Water Management System</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className=\"p-4 space-y-1\">
          <p className=\"px-4 py-2 text-xs font-semibold uppercase tracking-wider text-slate-500\">
            Main Menu
          </p>
          {mainItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              data-testid={`nav-${item.label.toLowerCase().replace(/ /g, '-')}`}
              onClick={() => setIsOpen(false)}
              className={({ isActive }) =>
                `sidebar-link ${isActive ? 'active' : ''}`
              }
            >
              <item.icon size={20} />
              <span>{item.label}</span>
            </NavLink>
          ))}
          
          <p className=\"px-4 py-2 mt-4 text-xs font-semibold uppercase tracking-wider text-slate-500\">
            Analysis Modules
          </p>
          {analysisItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              data-testid={`nav-${item.label.toLowerCase().replace(/ /g, '-')}`}
              onClick={() => setIsOpen(false)}
              className={({ isActive }) =>
                `sidebar-link ${isActive ? 'active' : ''}`
              }
            >
              <item.icon size={20} />
              <span className=\"text-sm\">{item.label}</span>
            </NavLink>
          ))}
        </nav>

        {/* Alert Summary */}
        <div className=\"absolute bottom-0 left-0 right-0 p-4 border-t border-slate-800 bg-slate-900\">
          <div className=\"bg-red-500/10 border border-red-500/30 rounded-lg p-3\">
            <div className=\"flex items-center gap-2 text-red-400 mb-1\">
              <AlertTriangle size={16} />
              <span className=\"text-sm font-semibold\">Active Alerts</span>
            </div>
            <p className=\"text-xs text-slate-400\">
              Critical drought warnings active in multiple districts
            </p>
          </div>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
"