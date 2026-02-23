                  </div>
                ) : weather ? (
                  <div className=\"space-y-2\">
                    <div className=\"flex items-center gap-2\">
                      <Thermometer className=\"w-4 h-4 text-orange-500\" />
                      <span className=\"text-sm text-slate-600\">{weather.temperature}°C</span>
                    </div>
                    <div className=\"flex items-center gap-2\">
                      <Droplets className=\"w-4 h-4 text-blue-500\" />
                      <span className=\"text-sm text-slate-600\">{weather.humidity}% humidity</span>
                    </div>
                    {weather.wind_speed && (
                      <div className=\"flex items-center gap-2\">
                        <span className=\"text-sm text-slate-600\">Wind: {weather.wind_speed} km/h</span>
                      </div>
                    )}
                    <p className=\"text-sm text-slate-500 capitalize\">{weather.description}</p>
                    <p className=\"text-xs text-blue-500 mt-2\">Source: {weather.source || 'open-meteo'}</p>
                  </div>
                ) : (
                  <p className=\"text-sm text-slate-400\">Weather data unavailable</p>
                )}
