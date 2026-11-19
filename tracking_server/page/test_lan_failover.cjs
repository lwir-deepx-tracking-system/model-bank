const test = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {join} = require('node:path');
const vm = require('node:vm');

const script = readFileSync(join(__dirname, 'js', '14_start_timers.js'), 'utf8');
const lan = 'http://192.168.42.10:8090/';

async function mock(settings = {}) {
  let clock = 1000;
  const origin = settings.origin || 'http://100.64.42.10:8090';
  const link = {hidden:true, addEventListener(){}, setAttribute(){}};
  const element = id => id === 'lan-web' ? link : {addEventListener(){}, setAttribute(){}};
  const redirects = [];
  const stored = new Map(settings.history || []);
  const responses = [settings.initial === undefined ? lan : settings.initial];
  const calls = [];
  const context = vm.createContext({
    $:element, rtick(){}, tick(){}, setInterval(){}, setTimeout(){return 1;}, clearTimeout(){},
    Date:{now:()=>clock}, lastOk:clock, grace:clock, URL, AbortController,
    localStorage:{getItem(){}}, sessionStorage:{getItem:k=>stored.get(k),setItem:(k,v)=>stored.set(k,v)},
    location:{origin,assign:url=>redirects.push(url)}, document:{hidden:false,body:{classList:{toggle(){}}}},
    dispatchEvent(){}, Event:class {},
    fetch:async (path,options)=>{
      calls.push({path,options});
      const answer = responses.shift();
      if(answer === undefined)throw new TypeError('connection down');
      return {ok:answer !== 503, json:async()=>({lan_url:answer})};
    }
  });
  vm.runInContext(script, context);
  await new Promise(setImmediate);
  return {context,link,redirects,stored,responses,calls,
    async go(ms){clock+=ms; await vm.runInContext('checkLanNetwork()', context);}
  };
}

test('opens the configured LAN page only after sustained loss and two failed checks', async ()=>{
  const f = await mock();
  assert.equal(f.link.href,lan);
  await f.go(4000);
  assert.equal(f.calls.length,1);
  await f.go(2000);
  assert.deepEqual(f.redirects,[]);
  await f.go(2000);
  assert.deepEqual(f.redirects,[lan]);
  await f.go(8000);
  assert.deepEqual(f.redirects,[lan]);
  assert.equal(f.stored.size,1);
});

test('no redirect while ordinary server replies keep arriving', async ()=>{
  const f = await mock();
  f.context.lastOk=5900;
  await f.go(5000);
  f.context.lastOk=10800;
  await f.go(5000);
  assert.deepEqual(f.redirects,[]);
  assert.equal(f.calls.length,1);
});

test('one failed check followed by a reachable server clears the failure count', async ()=>{
  const f = await mock();
  await f.go(6000);
  f.responses.push(lan);
  await f.go(2000);
  await f.go(6000);
  assert.deepEqual(f.redirects,[]);
  await f.go(2000);
  assert.deepEqual(f.redirects,[lan]);
});

test('HTTP errors prove the original server is reachable', async ()=>{
  const f = await mock();
  f.responses.push(503,503);
  await f.go(6000);
  await f.go(6000);
  assert.deepEqual(f.redirects,[]);
  assert.equal(f.link.href,lan);
});

test('a hidden tab or recent wake does not initiate a redirect', async ()=>{
  const f = await mock();
  f.context.document.hidden=true;
  await f.go(8000);
  f.context.document.hidden=false;
  f.context.grace=9000;
  await f.go(2000);
  assert.equal(f.calls.length,1);
  assert.deepEqual(f.redirects,[]);
});

test('the LAN page shows its active state and stays on its own address', async ()=>{
  const f = await mock({origin:new URL(lan).origin});
  await f.go(10000);
  assert.equal(f.link.textContent,'LAN 연결');
  assert.equal(f.calls.length,1);
  assert.deepEqual(f.redirects,[]);
});

test('unconfigured or invalid destinations are never used', async ()=>{
  for(const url of [null,'bad-url','http://8.8.8.8:8090/','http://wrong:8090/','http://u@10.0.0.1/','http://10.0.0.1/cmd']){
    const f=await mock({initial:url});
    await f.go(8000);
    await f.go(6000);
    assert.equal(f.link.hidden,true);
    assert.deepEqual(f.redirects,[]);
  }
});

test('Back remains usable after a failed LAN navigation', async ()=>{
  const f=await mock({history:[['lwir_lan_redirect_at','1000']]});
  await f.go(6000);
  await f.go(2000);
  assert.deepEqual(f.redirects,[]);
  await f.go(55000);
  assert.deepEqual(f.redirects,[lan]);
});
