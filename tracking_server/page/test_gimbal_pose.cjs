const test = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const vm = require('node:vm');

function setup() {
  const elements = {};
  const element = id => elements[id] ||= {
    hidden:true, attrs:{}, values:{}, naturalWidth:640, naturalHeight:480,
    clientWidth:640, clientHeight:480, addEventListener(){},
    setAttribute(key, value){this.attrs[key] = value;},
    querySelector(){return element(id+'-marker');},
    style:{setProperty(key, value){element(id).values[key] = value;}},
  };
  let now = 1000;
  const context = vm.createContext({$:element, addEventListener(){}, Date:{now:()=>now}});
  vm.runInContext(readFileSync(join(__dirname, 'js/gimbal_pose.js'), 'utf8'), context);
  return {context, element, advance:ms=>now+=ms};
}
function position(rig, axis) {return rig.element('pose-'+axis+'-marker').values['--pose-position'];}

test('both measured zero angles are exactly central', () => {
  const rig = setup();
  // Asymmetric hardware limits must not shift the zero centre.
  rig.context.renderGimbalPose({pan:{deg:0,min_deg:-180,max_deg:180},tilt:{deg:0,min_deg:-30,max_deg:90}});
  for (const axis of ['pan','tilt']) {
    assert.equal(position(rig, axis), '50%');
    assert.equal(rig.element('pose-'+axis).hidden, false);
    assert.match(rig.element('pose-'+axis).attrs['aria-label'], /0°/);
  }
});

test('pan -180/180 and tilt -90/90 are the two endpoints', () => {
  const rig = setup();
  for (const [sign, expected] of [[-1,'0%'],[1,'100%']]) {
    rig.context.renderGimbalPose({pan:{deg:sign*180},tilt:{deg:sign*90}});
    assert.equal(position(rig, 'pan'), expected);
    assert.equal(position(rig, 'tilt'), expected);
  }
});

test('intermediate angles have a symmetric scale', () => {
  const rig = setup();
  rig.context.renderGimbalPose({pan:{deg:-90},tilt:{deg:45}});
  assert.equal(position(rig, 'pan'), '25%');
  assert.equal(position(rig, 'tilt'), '75%');
});

test('missing measured feedback never substitutes motion estimates', () => {
  const rig = setup();
  rig.context.renderGimbalPose(null, {pan:50,tilt:50});
  assert.equal(rig.element('pose-pan').hidden, true);
  assert.equal(rig.element('pose-tilt').hidden, true);
});

test('invalid axes are hidden independently, display limits clamp without falsifying angle', () => {
  const rig = setup();
  rig.context.renderGimbalPose({pan:{deg:NaN},tilt:{deg:100}});
  assert.equal(rig.element('pose-pan').hidden, true);
  assert.equal(position(rig, 'tilt'), '100%');
  assert.match(rig.element('pose-tilt').attrs['aria-label'], /100°/);
  rig.context.renderGimbalPose({pan:{deg:180.05},tilt:{deg:0}});
  assert.equal(position(rig, 'pan'), '100%'); // encoder quantisation at the endpoint
  assert.match(rig.element('pose-pan').attrs['aria-label'], /180.05°/);
});

test('stale measurements disappear after 1.5 seconds', () => {
  const rig = setup();
  rig.context.renderGimbalPose({pan:{deg:20},tilt:{deg:-10}});
  rig.advance(1500);rig.context.expireGimbalPose();
  assert.equal(rig.element('pose-pan').hidden, true);
  assert.equal(rig.element('pose-tilt').hidden, true);
});
