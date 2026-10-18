import { useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Float, Line, OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';

const palette = { positive: '#59f4d6', warning: '#f8bd5a', danger: '#ff5c79', neutral: '#71a7ff' };

function Core({ tone, reducedMotion }) {
  const group = useRef();
  const ring = useRef();
  const color = palette[tone] || palette.neutral;
  useFrame((state, delta) => {
    if (!group.current || reducedMotion) return;
    group.current.rotation.y += delta * 0.2;
    group.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.7) * 0.08;
    if (ring.current) ring.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 2.1) * 0.06);
  });
  return (
    <group ref={group}>
      <mesh>
        <sphereGeometry args={[0.78, 40, 40]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={2.6} roughness={0.22} metalness={0.7} />
      </mesh>
      <mesh ref={ring} rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[1.02, 0.035, 16, 80]} />
        <meshBasicMaterial color={color} transparent opacity={0.9} />
      </mesh>
      <mesh rotation={[0.4, 0.2, 0.1]}>
        <torusGeometry args={[1.28, 0.014, 12, 64]} />
        <meshBasicMaterial color="#82b7ff" transparent opacity={0.4} />
      </mesh>
      <Text position={[0, -1.35, 0]} fontSize={0.13} color="#dce9ff" anchorX="center" letterSpacing={0.08}>
        RISK CORE
      </Text>
    </group>
  );
}

function EntityNode({ position, label, icon, color, active, onClick, reducedMotion }) {
  const mesh = useRef();
  useFrame((state) => {
    if (!mesh.current || reducedMotion) return;
    mesh.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 1.3 + position[0]) * 0.035;
  });
  return (
    <group position={position} onClick={(event) => { event.stopPropagation(); onClick(label); }}>
      <mesh ref={mesh}>
        <sphereGeometry args={[active ? 0.18 : 0.14, 20, 20]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={active ? 1.8 : 0.8} roughness={0.25} />
      </mesh>
      <Text position={[0, -0.3, 0]} fontSize={0.075} color={active ? '#ffffff' : '#8aa8d7'} anchorX="center" maxWidth={0.9}>
        {icon}  {label}
      </Text>
    </group>
  );
}

function SceneContent({ tone, selectedNode, onSelectNode, reducedMotion }) {
  const entityColor = palette[tone] || palette.neutral;
  const nodes = useMemo(() => [
    { position: [-2.25, 1.12, 0], label: 'Customer', icon: '◉' },
    { position: [2.25, 1.05, 0], label: 'Device', icon: '◌' },
    { position: [-2.25, -1.08, 0], label: 'Location', icon: '⌖' },
    { position: [2.25, -1.08, 0], label: 'Merchant', icon: '▦' }
  ], []);
  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[0, 0, 2.6]} intensity={8} color={entityColor} distance={7} />
      <pointLight position={[-3, 2, 1]} intensity={2.8} color="#6da8ff" distance={6} />
      <Float speed={reducedMotion ? 0 : 1.2} rotationIntensity={0.12} floatIntensity={0.16}>
        <Core tone={tone} reducedMotion={reducedMotion} />
      </Float>
      <Line points={[[-2.25, 1.12, 0], [-0.8, 0.45, 0], [0, 0, 0]]} color={entityColor} transparent opacity={0.65} lineWidth={1.1} />
      <Line points={[[2.25, 1.05, 0], [0.8, 0.45, 0], [0, 0, 0]]} color={entityColor} transparent opacity={0.65} lineWidth={1.1} />
      <Line points={[[-2.25, -1.08, 0], [-0.8, -0.45, 0], [0, 0, 0]]} color={entityColor} transparent opacity={0.48} lineWidth={1.1} />
      <Line points={[[2.25, -1.08, 0], [0.8, -0.45, 0], [0, 0, 0]]} color={entityColor} transparent opacity={0.48} lineWidth={1.1} />
      {nodes.map((node) => <EntityNode key={node.label} {...node} color={entityColor} active={selectedNode === node.label} onClick={onSelectNode} reducedMotion={reducedMotion} />)}
      <OrbitControls enablePan={false} enableZoom={false} autoRotate={!reducedMotion} autoRotateSpeed={0.22} minPolarAngle={Math.PI / 2.5} maxPolarAngle={Math.PI / 1.7} />
    </>
  );
}

function FallbackScene({ tone, selectedNode, onSelectNode }) {
  const color = palette[tone] || palette.neutral;
  return (
    <div className="scene-fallback" style={{ '--risk-color': color }}>
      <div className="fallback-grid" />
      <div className="fallback-core"><span>RISK</span><strong>CORE</strong></div>
      {['Customer', 'Device', 'Location', 'Merchant'].map((node, index) => (
        <button key={node} className={`fallback-node node-${index} ${selectedNode === node ? 'is-active' : ''}`} onClick={() => onSelectNode(node)}>
          <span>{['◉', '◌', '⌖', '▦'][index]}</span>{node}
        </button>
      ))}
      <div className="fallback-legend">WebGL fallback · select an entity to focus its evidence</div>
    </div>
  );
}

export default function RiskScene({ tone, selectedNode, onSelectNode, reducedMotion, webglAvailable = true }) {
  if (!webglAvailable) return <FallbackScene tone={tone} selectedNode={selectedNode} onSelectNode={onSelectNode} />;
  return <Canvas camera={{ position: [0, 0, 5.5], fov: 42 }} dpr={[1, 1.6]} gl={{ antialias: true, alpha: true }}>
    <SceneContent tone={tone} selectedNode={selectedNode} onSelectNode={onSelectNode} reducedMotion={reducedMotion} />
  </Canvas>;
}
