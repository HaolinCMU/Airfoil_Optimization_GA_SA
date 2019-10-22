function [pol,foil] = xfoilCl(coord, Cl, Re, Mach, varargin)
% Run XFoil and return the results.
% [polar,foil] = xfoil(coord,Cl,Re,Mach,{extra commands})
%
% Xfoil.exe needs to be in the same directory as this m function.
% For more information on XFoil visit these websites;
%  http://web.mit.edu/drela/Public/web/xfoil
% 
% Inputs:
%    coord: Normalised foil co-ordinates (n by 2 array, of x & y 
%           from the TE-top passed the LE to the TE bottom)
%           or a filename of the XFoil co-ordinate file
%           or a NACA 4 or 5 digit descriptor (e.g. 'NACA0012')
%    Cl: Coefficient of Lift, can be a vector for a polar
%       Re: Reynolds number (use Re=0 for inviscid mode)
%     Mach: Mach number
% extra commands: Extra XFoil commands
%           The extra XFoil commands need to be proper xfoil commands 
%           in a character array. e.g. 'oper/iter 150'
%
% The transition criterion Ncrit can be specified using the 
% 'extra commands' option as follows,
% foil = xfoil('NACA0012',10,1e6,0.2,'oper/vpar n 12')
%  
%   Situation           Ncrit
%   -----------------   -----
%   sailplane           12-14
%   motorglider         11-13
%   clean wind tunnel   10-12
%   average wind tunnel    9 <= standard "e^9 method"
%   dirty wind tunnel    4-8
%
% A flap deflection can be added using the following command,
% 'gdes flap {xhinge} {yhinge} {flap_defelction} exec' 
%
% Outputs:
%  polar: structure with the polar coefficients (alpha,CL,CD,CDp,CM,
%          Top_Xtr,Bot_Xtr) 
%   foil: stucture with the specific aoa values (s,x,y,UeVinf,
%          Dstar,Theta,Cf,H,cpx,cp) each column corresponds to a different
%          angle-of-attack.
%         If only one left hand operator is specified, only the polar will be parsed and output
%
% If there are different sized output arrays for the different incidence
% angles then they will be stored in a structured array, foil(1),foil(2)...
%
% If the output array does not have all alphas in it, that indicates a convergence failure in Xfoil.
% In that event, increase the iteration count with 'oper iter ##;
%
% Examples:
%    % Single AoA with a different number of panels
%    [pol foil] = xfoil('NACA0012',1.2,1e6,0.0,'panels n 330')
%

% Some default values
if ~exist('coord','var'), coord = 'NACA0012'; end;
if ~exist('Cl','var'), Cl = 0;    end;
if ~exist('Re','var'),    Re = 1e6;      end;
if ~exist('Mach','var'),  Mach = 0.2;    end;
Nalpha = length(Cl); % Number of alphas swept
% default foil name
foil_name = mfilename; 

% default filenames
wd = fileparts(which(mfilename)); % working directory, where xfoil.exe needs to be
fname = mfilename;
fname = fname(1:end-2);
file_coord= [foil_name '.foil'];

% Save coordinates
if ischar(coord),  % Either a NACA string or a filename
  if isempty(regexpi(coord,'^NACA *[0-9]{4,5}$')) % Check if a NACA string
%     foil_name = coord; % some redundant code removed to go green ( ~isempty if uncommented)
%   else             % Filename supplied
    % set coord file
    file_coord = coord;
  end;  
else
  % Write foil ordinate file
  if exist(file_coord,'file'),  delete(file_coord); end;
  pause(0.1)
  fid = fopen(file_coord,'w');
  if (fid<=0),
    error([mfilename ':io'],'Unable to create file %s',file_coord);
  else
    fprintf(fid,'%s\n',foil_name);
    fprintf(fid,'%9.5f   %9.5f\n',coord');
    fclose(fid);
  end;
end;

% Write xfoil command file
fid = fopen([wd filesep fname '.inp'],'w');
if (fid<=0),
  error([mfilename ':io'],'Unable to create xfoil.inp file');
else
  if ischar(coord),
    if ~isempty(regexpi(coord,'^NACA *[0-9]{4,5}$')),  % NACA string supplied
      fprintf(fid,'naca %s\n',coord(5:end));
    else  % filename supplied
      fprintf(fid,'load %s\n',file_coord);
    end;  
  else % Coordinates supplied, use the default filename
    fprintf(fid,'load %s\n',file_coord);
  end;
  % Extra Xfoil commands
  for ii = 1:length(varargin),
    txt = varargin{ii};
    txt = regexprep(txt,'[ \\\/]+','\n');
    fprintf(fid,'%s\n\n',txt);
  end;
  fprintf(fid,'\n\noper\n');
  % set Reynolds and Mach
  fprintf(fid,'re %g\n',Re);
  fprintf(fid,'mach %g\n',Mach);
  
  % Switch to viscous mode
  if (Re>0)
    fprintf(fid,'visc\n');  
  end;

  % Polar accumulation 
  fprintf(fid,'pacc\n\n\n');
  % Xfoil alpha calculations
  [file_dump, file_cpwr] = deal(cell(1,Nalpha)); % Preallocate cell arrays
  
  for ii = 1:Nalpha
    % Individual output filenames
    file_dump{ii} = sprintf('%s_c%06.3f_dump.dat',fname,Cl(ii));
    file_cpwr{ii} = sprintf('%s_c%06.3f_cpwr.dat',fname,Cl(ii));
    % Commands
    fprintf(fid,'c %g\n',Cl(ii));
    fprintf(fid,'dump %s\n',file_dump{ii});
    fprintf(fid,'cpwr %s\n',file_cpwr{ii});
  end;
  % Polar output filename
  file_pwrt = sprintf('%s_pwrt.dat',fname);
  fprintf(fid,'pwrt\n%s\n',file_pwrt);
  fprintf(fid,'plis\n');
  fprintf(fid,'\nquit\n');
  fclose(fid);

  % execute xfoil
  cmd = sprintf('cd %s && xfoil.exe < xfoil.inp > xfoil.out',wd);
  [status,result] = system(cmd);
  if (status~=0),
    disp(result);
    error([mfilename ':system'],'Xfoil execution failed! %s',cmd);
  end;

  % Read dump file
  %    #    s        x        y     Ue/Vinf    Dstar     Theta      Cf       H
  jj = 0;
  ind = 1;
% Note that 
foil.alpha = zeros(1,Nalpha); % Preallocate alphas
% Find the number of panels with an inital run
only = nargout; % Number of outputs checked. If only one left hand operator then only do polar

if only >1 % Only do the foil calculations if more than one left hand operator is specificed
  for ii = 1:Nalpha
    jj = jj + 1;

    fid = fopen([wd filesep file_dump{ii}],'r');
    if (fid<=0),
      error([mfilename ':io'],'Unable to read xfoil output file %s',file_dump{ii});
    else
      D = textscan(fid,'%f%f%f%f%f%f%f%f','Delimiter',' ','MultipleDelimsAsOne',true,'CollectOutput',1,'HeaderLines',1);
      fclose(fid);
      delete([wd filesep file_dump{ii}]);
      
      if ii == 1 % Use first run to determine number of panels (so that NACA airfoils work without vector input)
         Npanel = length(D{1}); % Number of airfoil panels pulled from the first angle tested
         % Preallocate Outputs
         [foil.s, foil.x, foil.y, foil.UeVinf, foil.Dstar, foil.Theta, foil.Cf, foil.H] = deal(zeros(Npanel,Nalpha));  
      end
      
      % store data
      if ((jj>1) && (size(D{1},1)~=length(foil(ind).x)) && sum(abs(foil(ind).x(:,1)-size(D{1},1)))>1e-6 ),
        ind = ind + 1;
        jj = 1;
      end;
      foil.s(:,jj) = D{1}(:,1);
      foil.x(:,jj) = D{1}(:,2);
      foil.y(:,jj) = D{1}(:,3);
      foil.UeVinf(:,jj) = D{1}(:,4);
      foil.Dstar(:,jj) = D{1}(:,5);
      foil.Theta(:,jj) = D{1}(:,6);
      foil.Cf(:,jj) = D{1}(:,7);
      foil.H (:,jj)= D{1}(:,8);
    end;

    foil.alpha(1,jj) = Cl(jj);

    % Read cp file
    fid = fopen([wd filesep file_cpwr{ii}],'r');
    if (fid<=0),
      error([mfilename ':io'],'Unable to read xfoil output file %s',file_cpwr{ii});
    else
      C = textscan(fid, '%10f%9f%f', 'Delimiter', '', 'WhiteSpace', '', 'HeaderLines' ,3, 'ReturnOnError', false);
      fclose(fid);
      delete([wd filesep file_cpwr{ii}]);
      % store data
      if ii == 1 % Use first run to determine number of panels (so that NACA airfoils work without vector input)
         NCp = length(C{1}); % Number of points Cp is listed for pulled from the first angle tested
         % Preallocate Outputs
         [foil.xcp, foil.cp] = deal(zeros(NCp,Nalpha));  
         foil.xcp = C{1}(:,1);
      end      
      foil.cp(:,jj) = C{3}(:,1);
    end;
  end;
end

if only <= 1% clear files for default run
  for ii=1:Nalpha % Clear out the xfoil dump files not used
      delete([wd filesep file_dump{ii}]);
      delete([wd filesep file_cpwr{ii}]);
  end 
end

  % Read polar file
  %  
  %       XFOIL         Version 6.96
  %  
  % Calculated polar for: NACA 0012                                       
  %  
  % 1 1 Reynolds number fixed          Mach number fixed         
  %  
  % xtrf =   1.000 (top)        1.000 (bottom)  
  % Mach =   0.000     Re =     1.000 e 6     Ncrit =  12.000
  %  
  %   alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr
  %  ------ -------- --------- --------- -------- -------- --------
  fid = fopen([wd filesep file_pwrt],'r');
  if (fid<=0),
    error([mfilename ':io'],'Unable to read xfoil polar file %s',file_pwrt);
  else
    % Header
    % Calculated polar for: NACA 0012 
    P = textscan(fid,' Calculated polar for: %[^\n]','Delimiter',' ','MultipleDelimsAsOne',true,'HeaderLines',3);
    pol.name = strtrim(P{1}{1});
    % xtrf =   1.000 (top)        1.000 (bottom)  
    P = textscan(fid, '%*s%*s%f%*s%f%s%s%s%s%s%s', 1, 'Delimiter', ' ', 'MultipleDelimsAsOne', true, 'HeaderLines', 2, 'ReturnOnError', false);
    pol.xtrf_top = P{1}(1);
    pol.xtrf_bot = P{2}(1);
    % Mach =   0.000     Re =     1.000 e 6     Ncrit =  12.000
    P = textscan(fid, '%*s%*s%f%*s%*s%f%*s%f%*s%*s%f', 1, 'Delimiter', ' ', 'MultipleDelimsAsOne', true, 'HeaderLines', 0, 'ReturnOnError', false);
    pol.Re = P{2}(1) * 10^P{3}(1);
    pol.Ncrit = P{4}(1);

    % data
    P = textscan(fid, '%f%f%f%f%f%f%f%*s%*s%*s%*s', 'Delimiter',  ' ', 'MultipleDelimsAsOne', true, 'HeaderLines' , 4, 'ReturnOnError', false);
    fclose(fid);
    delete([wd filesep file_pwrt]);
    % store data
    pol.alpha = P{1}(:,1);
    pol.CL  = P{2}(:,1);
    pol.CD  = P{3}(:,1);
    pol.CDp = P{4}(:,1);
    pol.Cm  = P{5}(:,1);
    pol.Top_xtr = P{6}(:,1);
    pol.Bot_xtr = P{7}(:,1);
  end
%   if length(pol.alpha) ~= Nalpha % Check if xfoil failed to converge 
%      warning('One or more Cl values failed to converge. Last converged was alpha = %f. Rerun with ''oper iter ##'' command.\n', pol.alpha(end)) 
%   end
  
end
